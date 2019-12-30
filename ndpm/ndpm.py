from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from .expert import Expert
from .priors import CumulativePrior


class Ndpm(nn.Module):
    def __init__(self, config, writer: SummaryWriter):
        super().__init__()
        self.config = config
        self.experts = nn.ModuleList([Expert(config)])
        self.stm_capacity = config['stm_capacity']
        self.stm_x = []
        self.stm_y = []
        self.stm_next_erase = config['stm_erase_period']
        self.prior = CumulativePrior(config)
        self.device = config['device'] if 'device' in config else 'cuda'
        self.writer = writer

    def get_experts(self):
        return tuple(self.experts.children())

    def forward(self, x, return_assignments=False):
        if len(self.experts) == 1:
            raise RuntimeError('There\'s no expert to run on the input')
        x = x.to(self.device)
        log_evid = -self.experts[-1].g.collect_nll(x)[0]  # [B, 1+K]
        log_evid = log_evid[:, 1:].unsqueeze(2)  # [B, K, 1]
        log_prior = -self.prior.nl_prior()[1:]  # [K]
        log_prior -= torch.logsumexp(log_prior, dim=0)
        log_prior = log_prior.unsqueeze(0).unsqueeze(2)  # [1, K, 1]
        log_joint = log_prior + log_evid  # [B, K, 1]
        assignments = log_joint.squeeze(2).argmax(dim=1)  # [B]
        if not self.config['disable_d']:
            log_pred = self.experts[-1].d.collect_forward(x)  # [B, 1+K, C]
            log_pred = log_pred[:, 1:, :]  # [B, K, C]
            log_joint = log_joint + log_pred  # [B, K, C]

        log_joint = log_joint.logsumexp(dim=1).squeeze()  # [B,] or [B, C]
        return (log_joint, assignments) if return_assignments else log_joint

    def learn(self, x, y, step):
        summarize = step % self.config['summary_step'] == 0
        x, y = x.to(self.device), y.to(self.device)

        if self.config.get('send_to_stm_always'):
            self.stm_x.extend(torch.unbind(x.cpu()))
            self.stm_y.extend(torch.unbind(y.cpu()))
        else:
            # Determine the destination of each data point
            nll, summaries = self.experts[-1].collect_nll(x, y)  # [B, 1+K]
            nl_prior = self.prior.nl_prior()  # [1+K]
            nl_joint = nll + nl_prior.unsqueeze(0).expand(
                nll.size(0), -1)  # [B, 1+K]

            if summarize:
                for i, summary in enumerate(summaries):
                    summary.write(self.writer, step, postfix='/{}'.format(i))

                mean_nl_joint = nl_joint.detach().mean(dim=0)
                if len(self.experts) > 1:
                    nl_joint_ndpm = nl_joint.detach()[:, 1:].min(dim=1)[0]
                    self.writer.add_scalar(
                        'nl_joint/ndpm/max', nl_joint_ndpm.max(dim=0)[0], step)
                    self.writer.add_scalar(
                        'nl_joint/ndpm/mean', nl_joint_ndpm.mean(dim=0), step)
                for k, expert in enumerate(self.experts):
                    self.writer.add_scalar(
                        'data counts per expert/%s' % expert.id,
                        self.prior.counts[k], step)
                    self.writer.add_scalar(
                        'nl_prior/%s' % expert.id, nl_prior[k], step)
                    self.writer.add_scalar(
                        'nl_joint/%s' % expert.id, mean_nl_joint[k], step)
                    self.writer.add_histogram(
                        'nl_joint_dist/%s' % expert.id, nl_joint[:, k], step)
                    self.writer.add_scalar(
                        'nl_cond/%s' % expert.id,
                        mean_nl_joint[k] - nl_prior[k], step)
                    self.writer.add_histogram(
                        'nl_cond_dist/%s' % expert.id,
                        nl_joint[:, k] - nl_prior[k], step)

            # Save to short-term memory
            destination = torch.argmin(nl_joint, dim=1).to(self.device)  # [B]
            if 'known_destination' in self.config:
                destination = torch.tensor(self.config['known_destination'])[y]
                destination = torch.where(
                    destination >= len(self.experts),
                    torch.zeros((), dtype=torch.int64), destination)
                destination = destination.to(self.device)
            to_stm = destination == 0  # [B]
            self.stm_x.extend(torch.unbind(x[to_stm].cpu()))
            self.stm_y.extend(torch.unbind(y[to_stm].cpu()))
            self.stm_next_erase -= 1
            if self.stm_next_erase == 0 and self.config['stm_erase_period'] > 0:
                if len(self.stm_x) > 0:
                    self.stm_x.pop(0)
                    self.stm_y.pop(0)
                self.stm_next_erase = self.config['stm_erase_period']

            # Train expert
            with torch.no_grad():
                min_joint = nl_joint.min(dim=1)[0].view(-1, 1)
                to_expert = torch.exp(-nl_joint + min_joint)  # [B, 1+K]
                to_expert[:, 0] = 0.  # [B, 1+K]
                to_expert = \
                    to_expert / (to_expert.sum(dim=1).view(-1, 1) + 1e-7)

            if 'known_destination' in self.config:
                to_expert = torch.eye(len(self.experts))[destination]
                to_expert = to_expert.to(self.device)

            # Compute losses per expert
            nll_for_train = nll * (1. - to_stm.float()).unsqueeze(1)  # [B,1+K]
            losses = (nll_for_train * to_expert).sum(0)  # [1+K]

            # Record expert usage
            expert_usage = to_expert.sum(dim=0)  # [K+1]
            self.prior.record_usage(expert_usage)

            # Do lr_decay implicitly
            if self.config['implicit_lr_decay']:
                losses = losses \
                    * self.config['stm_capacity'] / (self.prior.counts + 1e-8)
            loss = losses.sum()

            if loss.requires_grad:
                if 'update_min_usage' in self.config:
                    update_threshold = self.config['update_min_usage']
                else:
                    update_threshold = 0
                for k, usage in enumerate(expert_usage):
                    if usage > update_threshold:
                        self.experts[k].zero_grad()
                loss.backward()
                for k, usage in enumerate(expert_usage):
                    if usage > update_threshold:
                        self.experts[k].clip_grad()
                        self.experts[k].optimizer_step()
                        self.experts[k].lr_scheduler_step()

        # Sleep
        if len(self.stm_x) >= self.stm_capacity:
            dream_dataset = TensorDataset(
                torch.stack(self.stm_x), torch.stack(self.stm_y))
            self.sleep(dream_dataset)
            self.stm_x = []
            self.stm_y = []

    def sleep(self, dream_dataset):
        print('\nGoing to sleep...')
        # Add new expert and optimizer
        expert = Expert(self.config, self.get_experts())
        self.experts.append(expert)
        self.prior.add_expert()

        stacked_stm_x = torch.stack(self.stm_x)
        stacked_stm_y = torch.stack(self.stm_y)
        indices = torch.randperm(stacked_stm_x.size(0))
        train_size = stacked_stm_x.size(0) - self.config['sleep_val_size']
        dream_dataset = TensorDataset(
            stacked_stm_x[indices[:train_size]],
            stacked_stm_y[indices[:train_size]])
        dream_val_x = stacked_stm_x[indices[train_size:]]
        dream_val_y = stacked_stm_y[indices[train_size:]]

        # Prepare data iterator
        self.prior.record_usage(len(dream_dataset), index=-1)
        dream_iterator = iter(DataLoader(
            dream_dataset,
            batch_size=self.config['sleep_batch_size'],
            num_workers=self.config['sleep_num_workers'],
            sampler=RandomSampler(
                dream_dataset,
                replacement=True,
                num_samples=(
                        self.config['sleep_step_g'] *
                        self.config['sleep_batch_size']
                ))
        ))

        # Train generative component
        for step, (x, y) in enumerate(dream_iterator):
            step += 1
            x, y = x.to(self.device), y.to(self.device)
            g_loss, g_summary = expert.g.nll(x, y, step=step)
            g_loss = (g_loss + self.config['weight_decay']
                      * expert.g.weight_decay_loss())
            expert.g.zero_grad()
            g_loss.mean().backward()
            expert.g.clip_grad()
            expert.g.optimizer.step()

            if step % self.config['sleep_summary_step'] == 0:
                g_summary.write(
                    self.writer, step,
                    prefix='sleep_g_', postfix='/{}'.format(expert.id))
                print('\r   [Sleep-G %6d] loss: %5.1f' % (
                    step, g_loss.mean()
                ), end='')

                if self.config['sleep_val_size'] != 0:
                    with torch.no_grad():
                        val_loss_g, val_summary_g = expert.g.nll(
                            dream_val_x)
                    val_summary_g.add_tensor_summary(
                        'loss/total', val_loss_g, 'histogram')
                    val_summary_g.write(self.writer, step,
                                        prefix='sleep_val_g_',
                                        postfix='/{}'.format(expert.id))
        print()

        dream_iterator = iter(DataLoader(
            dream_dataset,
            batch_size=self.config['sleep_batch_size'],
            num_workers=self.config['sleep_num_workers'],
            sampler=RandomSampler(
                dream_dataset,
                replacement=True,
                num_samples=(
                        self.config['sleep_step_d'] *
                        self.config['sleep_batch_size'])
            )
        ))

        # Train discriminative component
        if not self.config['disable_d']:
            for step, (x, y) in enumerate(dream_iterator):
                step += 1
                x, y = x.to(self.device), y.to(self.device)
                d_loss, d_summary = expert.d.nll(x, y, step=step)
                d_loss = (d_loss + self.config['weight_decay']
                          * expert.d.weight_decay_loss())
                expert.d.zero_grad()
                d_loss.mean().backward()
                expert.d.clip_grad()
                expert.d.optimizer.step()

                if step % self.config['sleep_summary_step'] == 0:
                    d_summary.write(
                        self.writer, step,
                        prefix='sleep_d_', postfix='/{}'.format(expert.id))
                    print('\r   [Sleep-D %6d] loss: %5.1f' % (
                        step, d_loss.mean()
                    ), end='')

                    # Accuracy
                    with torch.no_grad():
                        pred = expert(x).argmax(1)
                    acc = (pred == y).float().mean()
                    self.writer.add_scalar(
                        'sleep_d_accuracy/%d' % expert.id, acc, step)

                    if self.config['sleep_val_size'] != 0:
                        with torch.no_grad():
                            val_loss_d, val_summary_d = expert.d.nll(
                                dream_val_x, dream_val_y)
                        val_summary_d.add_tensor_summary(
                            'loss/total', val_loss_d.mean(), 'scalar')
                        val_summary_d.write(self.writer, step,
                                            prefix='sleep_val_d',
                                            postfix='/{}'.format(expert.id))

                        # Validation accuracy
                        with torch.no_grad():
                            pred = expert(dream_val_x).argmax(1)
                        acc = (pred.cpu() == dream_val_y).float().mean()
                        self.writer.add_scalar(
                            'sleep_val_d_accuracy/%d' % expert.id, acc, step)

        expert.lr_scheduler_step()
        expert.lr_scheduler_step()
        expert.eval()

        print()

    @staticmethod
    def _nl_joint(nl_prior, nll):
        batch = nll.size(0)
        nl_prior = nl_prior.unsqueeze(0).expand(batch, -1)  # [B, 1+K]
        return nll + nl_prior

    def train(self, mode=True):
        # Disabled
        pass
