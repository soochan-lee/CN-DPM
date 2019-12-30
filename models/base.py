from abc import ABC, abstractmethod
from tensorboardX import SummaryWriter
from torch import nn as nn


# ==========
# Model ABCs
# ==========

class Model(nn.Module, ABC):
    def __init__(self, config, writer: SummaryWriter):
        super().__init__()
        self.config = config
        self.device = config['device']
        self.writer = writer

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def learn(self, x, y, t, step):
        pass
