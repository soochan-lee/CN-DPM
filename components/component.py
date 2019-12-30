from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import Tensor, autograd
from ndpm.summaries import Summaries, VaeSummaries, ClassificationSummaries
from typing import Tuple, List


class Component(nn.Module, ABC):
    def __init__(self, config, experts: Tuple):
        super().__init__()
        self.config = config
        self.device = config['device'] if 'device' in config else 'cuda'

        self.experts = experts

        self.optimizer = NotImplemented
        self.lr_scheduler = NotImplemented

    @abstractmethod
    def nll(self, x, y, step=None) -> Tuple[Tensor, Summaries]:
        """Return NLL"""
        pass

    @abstractmethod
    def collect_nll(self, x, y, step=None) -> Tuple[Tensor, List[Summaries]]:
        """Return NLLs including previous experts"""
        pass

    def _clip_grad_value(self, clip_value):
        for group in self.optimizer.param_groups:
            nn.utils.clip_grad_value_(group['params'], clip_value)

    def _clip_grad_norm(self, max_norm, norm_type=2):
        for group in self.optimizer.param_groups:
            nn.utils.clip_grad_norm_(group['params'], max_norm, norm_type)

    def clip_grad(self):
        clip_grad_config = self.config['clip_grad']
        if clip_grad_config['type'] == 'value':
            self._clip_grad_value(**clip_grad_config['options'])
        elif clip_grad_config['type'] == 'norm':
            self._clip_grad_norm(**clip_grad_config['options'])
        else:
            raise ValueError('Invalid clip_grad type: {}'
                             .format(clip_grad_config.type))

    @staticmethod
    def build_optimizer(optim_config, params):
        return getattr(torch.optim, optim_config['type'])(
            params, **optim_config['options'])

    @staticmethod
    def build_lr_scheduler(lr_config, optimizer):
        return getattr(torch.optim.lr_scheduler, lr_config['type'])(
            optimizer, **lr_config['options'])

    def weight_decay_loss(self):
        loss = torch.zeros([], device=self.device)
        for param in self.parameters():
            loss += torch.norm(param) ** 2
        return loss


class ComponentG(Component, ABC):
    def setup_optimizer(self):
        self.optimizer = self.build_optimizer(
            self.config['optimizer_g'], self.parameters())
        self.lr_scheduler = self.build_lr_scheduler(
            self.config['lr_scheduler_g'], self.optimizer)

    def collect_nll(self, x, y=None, step=None):
        """Default `collect_nll`

        Warning: Parameter-sharing components should implement their own
            `collect_nll`

        Returns:
            nll: Tensor of shape [B, 1+K]
            summaries: List of Summary
        """
        outputs = [expert.g.nll(x, y, step) for expert in self.experts]
        nll, summaries = [list(out) for out in zip(*outputs)]
        output = self.nll(x, y, step)
        nll.append(output[0])
        summaries.append(output[1])
        return torch.stack(nll, dim=1), summaries

    class Placeholder(Component):
        def forward(self, x):
            pass

        def nll(self, x, y=None, step=None) -> (Tensor, Summaries):
            pnll = self.config['placeholder_nll']
            nll = torch.ones([x.size(0)], device=self.device) * pnll
            summaries = VaeSummaries(
                loss_recon=torch.ones([], device=self.device) * pnll,
                loss_kl=torch.zeros([], device=self.device),
                loss_vae=torch.ones([], device=self.device) * pnll,
            )
            return nll, summaries

        def collect_nll(self, x, y, step=None) \
                -> Tuple[Tensor, List[Summaries]]:
            nll, summaries = self.nll(x, y, step)
            return nll.unsqueeze(1), [summaries]


class ComponentD(Component, ABC):
    def setup_optimizer(self):
        self.optimizer = self.build_optimizer(
            self.config['optimizer_d'], self.parameters())
        self.lr_scheduler = self.build_lr_scheduler(
            self.config['lr_scheduler_d'], self.optimizer)

    def collect_forward(self, x):
        """Default `collect_forward`

        Warning: Parameter-sharing components should implement their own
            `collect_forward`

        Returns:
            output: Tensor of shape [B, 1+K, C]
        """
        outputs = [expert.d(x) for expert in self.experts]
        outputs.append(self.forward(x))
        return torch.stack(outputs, 1)

    def collect_nll(self, x, y, step=None):
        """Default `collect_nll`

        Warning: Parameter-sharing components should implement their own
            `collect_nll`

        Returns:
            nll: Tensor of shape [B, 1+K]
            summaries: List of Summary
        """
        outputs = [expert.d.nll(x, y, step) for expert in self.experts]
        nll, summaries = [list(out)for out in zip(*outputs)]
        output = self.nll(x, y, step)
        nll.append(output[0])
        summaries.append(output[1])
        return torch.stack(nll, dim=1), summaries

    class Placeholder(Component, ABC):
        def __init__(self, config):
            super().__init__(config)
            self.p = nn.Parameter(torch.zeros([]), requires_grad=False)
            self.optimizer = self.build_optimizer(
                self.config['optimizer_d'], self.parameters())
            if self.config['lr_scheduler_d']['type'] == 'LambdaLR':
                self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                    self.optimizer,
                    lr_lambda=(lambda step: 0 if step == 0 else 1 / step)
                )
            else:
                self.lr_scheduler = self.build_lr_scheduler(
                    self.config['lr_scheduler_d'], self.optimizer)

        def forward(self, x):
            return self.dummy_out.expand(x.size(0), -1)

        def nll(self, x, y, step=None) -> (Tensor, Summaries):
            summaries = ClassificationSummaries(
                loss_pred=torch.zeros([], device=self.device))
            return torch.zeros([x.size(0)], device=self.device), summaries

        def collect_nll(self, x, y, step=None) \
                -> Tuple[Tensor, List[Summaries]]:
            nll, summaries = self.nll(x, y, step)
            return nll.unsqueeze(1), [summaries]
