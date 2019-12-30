from tensorboardX import SummaryWriter
from .singleton_model import SingletonModel
import torch
import random


class ReservoirModel(SingletonModel):
    def __init__(self, config, writer: SummaryWriter):
        super().__init__(config, writer)
        self.rsvr_size = config['reservoir_size']
        self.rsvr_x, self.rsvr_y = [], []
        self.n = 0

    def learn(self, x, y, t, step=None):
        x, y = x.to(self.device), y.to(self.device)

        # Replay reservoir
        if len(self.rsvr_x) > 0:
            k = min(len(self.rsvr_x), x.size(0))
            replay_idx = random.sample(range(len(self.rsvr_x)), k=k)
            replay_x = [self.rsvr_x[i] for i in replay_idx]
            replay_y = [self.rsvr_y[i] for i in replay_idx]
            merged_x = torch.cat([x, torch.stack(replay_x, dim=0)], dim=0)
            merged_y = torch.cat([y, torch.stack(replay_y, dim=0)], dim=0)
        else:
            merged_x, merged_y = x, y
        nll, summary = self.component.nll(merged_x, merged_y, step=step)
        weight_decay = self.component.weight_decay_loss()
        self.component.zero_grad()
        (nll.mean() + self.config['weight_decay'] * weight_decay).backward()
        self.component.clip_grad()
        self.component.optimizer.step()
        self.component.lr_scheduler.step()

        # Update reservoir
        for i in range(x.size(0)):
            if self.n < self.rsvr_size:
                self.rsvr_x.append(x[i])
                self.rsvr_y.append(y[i])
            else:
                m = random.randrange(self.n)
                if m < self.rsvr_size:
                    self.rsvr_x[m] = x[i]
                    self.rsvr_y[m] = y[i]
            self.n += 1

        if step % self.config['summary_step'] == 0:
            summary.write(self.writer, step)
            grad = torch.cat([
                p.grad.view(-1)
                for p in self.component.parameters()
                if p.grad is not None
            ], dim=0)
            self.writer.add_histogram('grad', grad, step)
            self.writer.add_scalar(
                'num_params', sum([p.numel() for p in self.parameters()]),
                step
            )
