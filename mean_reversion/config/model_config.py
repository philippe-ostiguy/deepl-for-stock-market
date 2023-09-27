from torch.optim.lr_scheduler import (
    OneCycleLR,
    ExponentialLR,
    ReduceLROnPlateau,
)
from pytorch_forecasting.metrics import RMSE, QuantileLoss, MASE, NormalDistributionLoss, SMAPE,DistributionLoss
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

LR_SCHEDULER_MAPPING = {
    "OneCycleLR": OneCycleLR,
    "ExponentialLR": ExponentialLR,
    "ReduceLROnPlateau": ReduceLROnPlateau,
}

PYTORCH_CALLBACKS = {
    'EarlyStopping' : EarlyStopping,
    'ModelCheckPoint' : ModelCheckpoint

}

LOSS = {
    'RMSE' : RMSE,
    'QuantileLoss': QuantileLoss,
    'MASE' : MASE,
    'NormalDistributionLoss' : NormalDistributionLoss,
    'SMAPE' : SMAPE,
    'DistributionLoss' : DistributionLoss

}
