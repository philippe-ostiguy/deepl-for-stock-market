from torchmetrics import Metric
import torch
from pytorch_forecasting import TemporalFusionTransformer, DeepAR, NHiTS, RecurrentNetwork
from mean_reversion.config.config_utils import ConfigManager

class PortfolioReturnMetric(Metric):
    higher_is_better = True
    full_state_update = True
    def __init__(self, dist_sync_on_step=True, config_manager = ConfigManager()):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self._lower_index, self._upper_index = config_manager.get_confidence_indexes()
        self._config =config_manager.config
        self.add_state("portfolio_value", default=torch.tensor(0.0), dist_reduce_fx="sum")

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



        portfolio_value = 1.0
        no_position_count = 0
        for actual, low_pred, high_pred in zip( target,
                                                     low_predictions,
                                                     high_predictions):
            if low_pred > 0 and high_pred >0:
                portfolio_value *= (1 + actual)
            elif high_pred < 0 and low_pred < 0:
                portfolio_value *= (1 - actual)
            else:
               no_position_count +=1

        self.portfolio_value += portfolio_value-1


    def compute(self):
        return self.portfolio_value


class BaseReturnMetricModel:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.portfolio_metric = PortfolioReturnMetric()

    def validation_step(self, batch, batch_idx):
        log = super().validation_step(batch, batch_idx)

        prediction_batch, targets = batch
        preds = log.get("prediction", None)
        if preds is None:
            preds = self(prediction_batch)

        stock_return = self.portfolio_metric(preds, targets)
        log["portfolio_return"] = stock_return

        return log

    def on_validation_epoch_end(self):
        product_of_returns = torch.stack(
            [(1 + x['portfolio_return']) for x in self.validation_step_outputs]
        ).prod()

        compounded_return = product_of_returns - 1
        self.log('val_PortfolioReturnMetric', compounded_return)
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
