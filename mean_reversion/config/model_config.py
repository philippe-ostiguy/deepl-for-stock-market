from pytorch_forecasting import TemporalFusionTransformer, NHiTS,DeepAR,RecurrentNetwork
from torch.optim.lr_scheduler import (
    OneCycleLR,
    ExponentialLR,
    ReduceLROnPlateau,
)
from pytorch_forecasting.metrics import RMSE, QuantileLoss, MASE, NormalDistributionLoss, SMAPE
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from mean_reversion.config.model_customizer import (
    PortfolioReturnMetric,
    CustomTemporalFusionTransformer,
    CustomDeepAR,
    CustomlRecurrentNetwork,
    CustomNHiTS
)

# models
MODEL_MAPPING = {
    "TemporalFusionTransformer": TemporalFusionTransformer,
    "NHiTS": NHiTS,
    "DeepAR": DeepAR,
    "RecurrentNetwork" : RecurrentNetwork
}
CUSTOM_MODEL = {
    "DeepAR" : CustomDeepAR,
    "TemporalFusionTransformer": CustomTemporalFusionTransformer,
    "NHiTS": CustomNHiTS,
    "RecurrentNetwork": CustomlRecurrentNetwork

}

LR_SCHEDULER_MAPPING = {
    "OneCycleLR": OneCycleLR,
    "ExponentialLR": ExponentialLR,
    "ReduceLROnPlateau": ReduceLROnPlateau,
}

PYTORCH_CALLBACKS = {
    'EarlyStopping' : EarlyStopping,
    'ModelCheckPoint' : ModelCheckpoint

}

TORCH_METRICS = {
    'PortfolioReturnMetric' : PortfolioReturnMetric
}

LOSS = {
    'RMSE' : RMSE,
    'QuantileLoss': QuantileLoss,
    'MASE' : MASE,
    'NormalDistributionLoss' : NormalDistributionLoss,
    'SMAPE' : SMAPE,

}
