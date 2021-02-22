import torch
import torch.nn as nn

from common.train.callbacks import Callback
from common.train.trainer import Trainer


class WeightClampCallback(Callback):

    def __init__(self, model: nn.Module, max_weights_value: float = -1):
        self.model = model
        self.max_weights_value = max_weights_value

    def on_epoch_train_end(self, trainer: Trainer, metric_values):
        if self.max_weights_value > 0:
            with torch.no_grad():
                for param in self.model.parameters():
                    param.data = torch.clamp(param.data, min=-self.max_weights_value, max=self.max_weights_value)
