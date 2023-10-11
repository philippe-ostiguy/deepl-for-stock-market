from torchmetrics import Metric
import torch
from pytorch_forecasting import TemporalFusionTransformer, DeepAR, NHiTS, RecurrentNetwork
from mean_reversion.config.config_utils import ConfigManager, ModelValueRetriver


class PortfolioReturnMetric(Metric):
    higher_is_better = True
    full_state_update = True
    def __init__(self, dist_sync_on_step=True, values_retriever = ModelValueRetriver(), config_manager = ConfigManager()):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self._lower_index, self._upper_index = values_retriever.confidence_indexes
        self._config =config_manager.config
        self.add_state("daily_returns", default=torch.tensor([]), dist_reduce_fx="cat")

    def update(self, preds: torch.Tensor, target_tensor: tuple):
        values = preds['prediction'].detach().cpu().numpy().squeeze()
        preds = values.tolist()

        if all(isinstance(lst, list) for lst in preds):
            sorted_preds = [sorted(lst) for lst in preds]
            low_predictions = [lst[self._lower_index] for lst in sorted_preds]
            high_predictions =[lst[self._upper_index] for lst in sorted_preds]
        else :
            low_predictions = preds
            high_predictions = preds

        target_tensor = target_tensor[0]
        target = target_tensor.squeeze().tolist()


        if not self._config["common"]["make_data_stationary"]:
            former_target = target
            target = [(target[i] / target[i - 1]) -1 for i in range(1, len(target))]

            low_predictions = [(low_predictions[i] / former_target[i - 1]) -1 for i in
                               range(1, len(low_predictions))]
            high_predictions = [(high_predictions[i] / former_target[i - 1]) -1 for i in
                                range(1, len(high_predictions))]

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
        self.daily_returns = torch.cat(
            [self.daily_returns, daily_returns_tensor], dim=0)
        print(f'\nNb of trade in update(): {self.daily_returns.shape[0]}')


    def compute(self):

        if self.daily_returns.shape[0] == 0 or not self.daily_returns.shape[0]:
            return torch.tensor(0.0)

        if self.daily_returns.shape[0] == 0:
            return torch.tensor(0.0)


        annualized_return = torch.prod(1 + self.daily_returns) ** (
                    252.0 / self.daily_returns.shape[0]) - 1
        annualized_risk = self.daily_returns.std() * (252 ** 0.5)
        return_on_risk = annualized_return / annualized_risk if annualized_risk != 0 else torch.tensor(
            0.0)
        print(f'\nReturn on risk in compute(): {return_on_risk.item()}')

        return return_on_risk


class BaseReturnMetricModel:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.portfolio_metric = PortfolioReturnMetric()
        self.best_return_on_risk = None
        self.best_epoch = None

    def validation_step(self, batch, batch_idx):
        log = super().validation_step(batch, batch_idx)

        prediction_batch, targets = batch
        preds = log.get("prediction", None)
        if preds is None:
            preds = self(prediction_batch)

        return_on_risk = self.portfolio_metric(preds, targets)
        self.log('return_on_risk', return_on_risk, on_step=True, on_epoch=True)

        return self.log

    def on_validation_epoch_end(self):
        aggregated_return_on_risk = self.trainer.callback_metrics[
            'return_on_risk']

        self.log('val_PortfolioReturnMetric', aggregated_return_on_risk)


        if self.best_return_on_risk is None or aggregated_return_on_risk > self.best_return_on_risk:
            self.best_return_on_risk = aggregated_return_on_risk
            self.best_epoch = self.current_epoch

        print(
            f"\nBest Return on Risk so far: {self.best_return_on_risk}, achieved at epoch: {self.best_epoch}")

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
