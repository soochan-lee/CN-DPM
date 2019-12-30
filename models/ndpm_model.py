import yaml
from tensorboardX import SummaryWriter
import torch
from torch import nn
from ndpm import Ndpm
from .base import Model


class NdpmModel(Model):
    def __init__(self, config, writer: SummaryWriter):
        super().__init__(config, writer)
        self.ndpm = Ndpm(config, writer)
        self.extractor = None

    def forward(self, x, expert_index=None, return_assignments=False):
        x = x.to(self.device)
        return (
            self.ndpm(x, return_assignments) if expert_index is None else
            self.ndpm.experts[expert_index](x)
        )

    def learn(self, x, y, t, step=None):
        x, y = x.to(self.device), y.to(self.device)
        self.ndpm.learn(x, y, step)

        if step % self.config['summary_step'] == 0:
            self.writer.add_scalar(
                'num_params', sum([
                    p.numel()
                    for e in self.ndpm.experts[1:]
                    for p in e.parameters()
                ]), step
            )
