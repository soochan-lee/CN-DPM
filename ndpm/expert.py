import torch
import torch.nn as nn
from .summaries import (
    Summaries,
)
from components.component import ComponentG, ComponentD
from components import G, D
from typing import Tuple


class Expert(nn.Module):
    def __init__(self, config, experts: Tuple = ()):
        super().__init__()
        self.config = config
        self.id = len(experts)
        self.experts = experts
        self.device = config['device'] if 'device' in config else 'cuda'

        self.g: ComponentG = G[config['g']](config, experts)
        self.d: ComponentD = (
            D[config['d']](config, experts) if not config['disable_d'] else None
        )

        # use random initialized g if it's a placeholder
        if self.id == 0:
            self.eval()
            for p in self.g.parameters():
                p.requires_grad = False

        # use random initialized d if it's a placeholder
        if self.id == 0 and self.d is not None:
            for p in self.d.parameters():
                p.requires_grad = False

        # otherwise use pretrained weights if provided
        if config.get('pretrained_init') is not None:
            state_dict = torch.load(config['pretrained_init'])
            state_dict = {
                k.split('component.')[1]: v for k, v in state_dict.items()
            }
            self.g.load_state_dict(state_dict)

    def forward(self, x):
        return self.d(x)

    def nll(self, x, y, step=None) -> (torch.Tensor, Summaries):
        """Negative log likelihood"""
        nll, summaries = self.g.nll(x, step)
        if self.d is not None:
            d_nll, d_summaries = self.d.nll(x, y, step)
            nll = nll + d_nll
            summaries = summaries + d_summaries
            summaries.add_tensor_summary('loss/total', nll.mean(), 'scalar')
        return nll, summaries

    def collect_nll(self, x, y, step=None):
        if self.id == 0:
            nll, summaries = self.nll(x, y, step)
            return nll.unsqueeze(1), [summaries]

        nll, summaries = self.g.collect_nll(x, step)
        if self.d is not None:
            d_nll, d_summaries = self.d.collect_nll(x, y, step)
            nll = nll + d_nll

            # Merge summaries
            new_summaries = []
            for i, (g_s, d_s) in enumerate(zip(summaries, d_summaries)):
                new_summary = g_s + d_s
                new_summary.add_tensor_summary(
                    'loss/total', nll[:, i].mean(), 'scalar')
                new_summaries.append(new_summary)
            summaries = new_summaries
        return nll, summaries

    def lr_scheduler_step(self):
        if self.g.lr_scheduler is not NotImplemented:
            self.g.lr_scheduler.step()
        if self.d is not None and self.d.lr_scheduler is not NotImplemented:
            self.d.lr_scheduler.step()

    def clip_grad(self):
        self.g.clip_grad()
        if self.d is not None:
            self.d.clip_grad()

    def optimizer_step(self):
        self.g.optimizer.step()
        if self.d is not None:
            self.d.optimizer.step()
