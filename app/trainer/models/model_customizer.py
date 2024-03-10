from torchmetrics import Metric
import torch
from pytorch_forecasting import TemporalFusionTransformer, DeepAR, NHiTS, RecurrentNetwork
from app.shared.config.config_utils import ConfigManager
from app.trainer.config.config_utils import ModelValueRetriver
from app.trainer.models.common import get_risk_rewards_metrics
import logging
from typing import Optional

class PortfolioReturnMetric(Metric):
    higher_is_better = True
    full_state_update = True
    update_count = 0

    def __init__(self, dist_sync_on_step=True, values_retriever = ModelValueRetriver(), config_manager : Optional [ConfigManager] = None):
        if config_manager is None:
            config_manager = ConfigManager(file='app/trainer/config.yaml')
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self._lower_index, self._upper_index = values_retriever.confidence_indexes
        self._config =config_manager.config
        self.add_state("daily_returns", default=[], dist_reduce_fx="cat")

    def update(self, preds: torch.Tensor, target_tensors: tuple):
        PortfolioReturnMetric.update_count += 1

        if PortfolioReturnMetric.update_count > 1:
            if self.daily_returns:
                raise ValueError("Daily returns should be an empty list or a tensor with value 0")

        adapted_target_tensor = target_tensors[0]
        targets_size = len(adapted_target_tensor)
        for item in range(targets_size):
            print(f'current item in update() : {item}')
            target_tensor = adapted_target_tensor[item]
            values = preds['prediction'][item].detach().cpu().numpy().squeeze()
            list_preds = values.tolist()
            if all(isinstance(lst, list) for lst in list_preds):
                low_predictions = [lst[self._lower_index] for lst in list_preds]
                high_predictions =[lst[self._upper_index] for lst in list_preds]
            else :
                low_predictions = list_preds
                high_predictions = list_preds
            target = target_tensor.squeeze().tolist()

            daily_returns = []
            no_position_count = 0
            for actual, low_pred, high_pred in zip( target,
                                                         low_predictions,
                                                         high_predictions):
                if low_pred > 0 and high_pred >0:
                    daily_returns.append(actual)
                elif high_pred < 0 and low_pred < 0:
                    daily_returns.append(-actual)
                else:
                   no_position_count +=1

            daily_returns_tensor = torch.tensor(daily_returns)
            if len(self.daily_returns) <= item:
                self.daily_returns.append(daily_returns_tensor)
            else:
                self.daily_returns[item] = torch.cat(
                    [self.daily_returns[item], daily_returns_tensor], dim=0)

            logging.warning(f'Nb of trade in update(): {len(self.daily_returns[item])}')


    def compute(self):
        if not self.daily_returns:
            return torch.tensor(0.0)

        weighted_return_on_risks = torch.tensor(0.0)
        total_trades = 0
        all_metrics = []

        for idx, returns in enumerate(self.daily_returns):
            print(f'Item index in compute: {idx}')
            metrics = get_risk_rewards_metrics(returns)
            num_of_trades = returns.shape[0]
            return_on_risk = metrics['return_on_risk']
            weighted_return_on_risks += return_on_risk * num_of_trades
            total_trades += num_of_trades
            all_metrics.append(metrics)

        if total_trades <= self._config['common']['min_nb_trades']*len(self.daily_returns):
            logging.warning(
                f'Low nb of trades in compute: {total_trades}')
            return torch.tensor(0.0)
        avg_weighted_return_on_risk = weighted_return_on_risks / total_trades
        if not torch.is_tensor(avg_weighted_return_on_risk):
            avg_weighted_return_on_risk = torch.tensor(
                avg_weighted_return_on_risk)

        logging.warning(
            f'Weighted return on risk in compute(): {avg_weighted_return_on_risk}')
        threshold_evaluation = 20
        if avg_weighted_return_on_risk > threshold_evaluation:
            logging.warning(f"On current epoch, avg_weighted_return_on_risk is "
                            f"higher than {threshold_evaluation}: {avg_weighted_return_on_risk}")
            return torch.tensor(0.0)
        return avg_weighted_return_on_risk


class BaseReturnMetricModel:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.portfolio_metric = PortfolioReturnMetric()
        self.best_return_on_risk = None
        self.best_epoch = None

    def validation_step(self, batch, batch_idx):
        log = super().validation_step(batch, batch_idx)
        batch_size = log['n_samples']
        prediction_batch, targets = batch
        preds = log.get("prediction", None)
        if preds is None:
            preds = self(prediction_batch)

        return_on_risk = self.portfolio_metric(preds, targets)
        self.log('return_on_risk', return_on_risk, on_step=True, on_epoch=True, batch_size=batch_size)

        return self.log

    def on_validation_epoch_end(self):
        aggregated_return_on_risk = self.trainer.callback_metrics[
            'return_on_risk']

        self.log('val_PortfolioReturnMetric', aggregated_return_on_risk)


        if self.best_return_on_risk is None or aggregated_return_on_risk > self.best_return_on_risk:
            self.best_return_on_risk = aggregated_return_on_risk
            self.best_epoch = self.current_epoch

        logging.warning(
            f"Best Return on Risk so far: {self.best_return_on_risk}, achieved at epoch: {self.best_epoch}")

        self.validation_step_outputs.clear()
        self.portfolio_metric.reset()
        super().on_validation_epoch_end()

class CustomTemporalFusionTransformer(BaseReturnMetricModel, TemporalFusionTransformer):
    pass

class CustomDeepAR(BaseReturnMetricModel, DeepAR):
    pass

class CustomlRecurrentNetwork(BaseReturnMetricModel, RecurrentNetwork):
    pass

class CustomNHiTS(BaseReturnMetricModel, NHiTS):
    pass
