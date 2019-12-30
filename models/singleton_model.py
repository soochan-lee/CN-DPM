from tensorboardX import SummaryWriter
from .base import Model
from components import G, D
from components.component import ComponentD
import torch


class SingletonModel(Model):
    def __init__(self, config, writer: SummaryWriter):
        super().__init__(config, writer)
        self.config = config
        self.device = config['device']
        self.writer = writer
        if config['g'] is not None:
            self.component = G[config['g']](config, ())
        elif config['d'] is not None:
            self.component = D[config['d']](config, ())
        else:
            raise RuntimeError('Component not specified.')

    def forward(self, x, expert_index=None):
        x = x.to(self.device)
        return (
            self.component(x) if isinstance(self.component, ComponentD) else
            -self.component.nll(x)[0]
        )

    def learn(self, x, y, t, step=None):
        x, y = x.to(self.device), y.to(self.device)
        nll, summary = self.component.nll(x, y, step=step)
        weight_decay = self.component.weight_decay_loss()
        self.component.zero_grad()
        (nll.mean() + self.config['weight_decay'] * weight_decay).backward()
        self.component.clip_grad()
        self.component.optimizer.step()
        self.component.lr_scheduler.step()

        if step % self.config['summary_step'] == 0:
            summary.write(self.writer, step)
            grad = torch.cat([
                p.grad.view(-1)
                for p in self.component.parameters()
                if p.grad is not None
            ], dim=0)
            self.writer.add_scalar(
                'lr', self.component.optimizer.param_groups[0]['lr'], step)
            self.writer.add_histogram('grad', grad, step)
            self.writer.add_scalar(
                'num_params', sum([p.numel() for p in self.parameters()]),
                step
            )
