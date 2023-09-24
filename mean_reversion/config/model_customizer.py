from torchmetrics import Metric
import torch
from pytorch_forecasting import TemporalFusionTransformer, DeepAR, NHiTS, RecurrentNetwork
from statistics import median

class PortfolioReturnMetric(Metric):
    def __init__(self, dist_sync_on_step=True):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("portfolio_value", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target_tensor: tuple):
        values = preds['prediction'].numpy().squeeze()

        preds = values.tolist()
        if all(isinstance(lst, list) for lst in preds):
            predictions = [median(lst) for lst in preds]
        else :
            predictions = preds

        target_tensor = target_tensor[0]

        target = target_tensor.squeeze().tolist()

        portfolio_value = 1.0
        for pred, actual in zip(predictions, target):
            if pred >= 0:
                portfolio_value *= (1 + actual)
            else:
                portfolio_value *= (1 - actual)
        self.portfolio_value = torch.tensor(self.portfolio_value.item() + 1) * portfolio_value


        self.portfolio_value-=1
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

class CustomTemporalFusionTransformer(BaseReturnMetricModel, TemporalFusionTransformer):
    pass

class CustomDeepAR(BaseReturnMetricModel, DeepAR):
    pass

class CustomlRecurrentNetwork(BaseReturnMetricModel, RecurrentNetwork):
    pass

class CustomNHiTS(BaseReturnMetricModel, NHiTS):
    pass
