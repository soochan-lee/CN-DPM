from abc import ABC, abstractmethod
from collections import Iterator, OrderedDict
from functools import reduce
import os
import math
import torch
from torch.utils.data import (
    Dataset,
    ConcatDataset,
    Subset,
    DataLoader,
    RandomSampler,
)
from torch.utils.data.dataloader import default_collate
import torchvision
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
from models.base import Model
from models.ndpm_model import NdpmModel


# =====================
# Base Classes and ABCs
# =====================


class DataScheduler(Iterator):
    def __init__(self, config):
        self.config = config
        self.schedule = config['data_schedule']
        self.datasets = OrderedDict()
        self.eval_datasets = OrderedDict()
        self.total_step = 0
        self.stage = -1

        # Prepare datasets
        for i, stage in enumerate(self.schedule):
            stage_total = 0
            for j, subset in enumerate(stage['subsets']):
                dataset_name, subset_name = subset
                if dataset_name in self.datasets:
                    stage_total += len(
                        self.datasets[dataset_name].subsets[subset_name])
                    continue
                self.datasets[dataset_name] = DATASET[dataset_name](config)
                self.eval_datasets[dataset_name] = DATASET[dataset_name](
                    config, train=False
                )
                stage_total += len(
                    self.datasets[dataset_name].subsets[subset_name])

            if 'steps' in stage:
                self.total_step += stage['steps']
            elif 'epochs' in stage:
                self.total_step += int(
                    stage['epochs'] * (stage_total // config['batch_size']))
                if stage_total % config['batch_size'] > 0:
                    self.total_step += 1
            elif 'steps' in stage:
                self.total_step += sum(stage['steps'])
            else:
                self.total_step += stage_total // config['batch_size']

        self.iterator = None

    def __next__(self):
        try:
            if self.iterator is None:
                raise StopIteration
            data = next(self.iterator)
        except StopIteration:
            # Progress to next stage
            self.stage += 1
            print('\nProgressing to stage %d' % self.stage)
            if self.stage >= len(self.schedule):
                raise StopIteration

            stage = self.schedule[self.stage]
            collate_fn = list(self.datasets.values())[0].collate_fn
            subsets = []
            for dataset_name, subset_name in stage['subsets']:
                subsets.append(
                    self.datasets[dataset_name].subsets[subset_name])
            dataset = ConcatDataset(subsets)

            # Determine sampler
            if 'samples' in stage:
                sampler = RandomSampler(
                    dataset,
                    replacement=True,
                    num_samples=stage['samples']
                )
            elif 'steps' in stage:
                sampler = RandomSampler(
                    dataset,
                    replacement=True,
                    num_samples=stage['steps'] * self.config['batch_size']
                )
            elif 'epochs' in stage:
                sampler = RandomSampler(
                    dataset,
                    replacement=True,
                    num_samples=(int(stage['epochs'] * len(dataset))
                                 + len(dataset) % self.config['batch_size'])
                )
            else:
                sampler = RandomSampler(dataset)

            self.iterator = iter(DataLoader(
                dataset,
                batch_size=self.config['batch_size'],
                num_workers=self.config['num_workers'],
                collate_fn=collate_fn,
                sampler=sampler,
                drop_last=True,
            ))

            data = next(self.iterator)

        # Get next data
        return data[0], data[1], self.stage

    def __len__(self):
        return self.total_step

    def eval(self, model, writer, step, eval_title):
        for i, eval_dataset in enumerate(self.eval_datasets.values()):
            # NOTE: we assume that each task is a dataset in multi-dataset
            # episode
            eval_dataset.eval(
                model, writer, step, eval_title,
                task_index=(i if len(self.eval_datasets) > 1 else None)
            )


class BaseDataset(Dataset, ABC):
    name = 'base'

    def __init__(self, config, train=True):
        self.config = config
        self.subsets = {}
        self.train = train

    def eval(self, model: Model, writer: SummaryWriter, step, eval_title,
             task_index=None):
        if self.config['eval_d']:
            self._eval_discriminative_model(model, writer, step, eval_title)
        if self.config['eval_g']:
            self._eval_generative_model(model, writer, step, eval_title)
        if 'eval_t' in self.config and self.config['eval_t']:
            self._eval_hard_assign(
                model, writer, step, eval_title,
                task_index=task_index
            )

    @abstractmethod
    def _eval_hard_assign(
            self,
            model,
            writer: SummaryWriter,
            step, eval_titlem, task_index=None):
        raise NotImplementedError

    @abstractmethod
    def _eval_discriminative_model(
            self,
            model,
            writer: SummaryWriter,
            step, eval_title):
        raise NotImplementedError

    @abstractmethod
    def _eval_generative_model(
            self,
            model: Model,
            writer: SummaryWriter,
            step, eval_title):
        raise NotImplementedError

    def collate_fn(self, batch):
        return default_collate(batch)


class CustomSubset(Subset):
    def __init__(self, dataset, indices, transform):
        super().__init__(dataset, indices)
        self.transform = transform

    def __getitem__(self, idx):
        return self.transform(self.dataset[self.indices[idx]])


# ================
# Generic Datasets
# ================

class ClassificationDataset(BaseDataset, ABC):
    num_classes = NotImplemented
    targets = NotImplemented

    def __init__(self, config, train=True):
        super().__init__(config, train)

    def _eval_hard_assign(
            self,
            model: NdpmModel,
            writer: SummaryWriter,
            step, eval_title, task_index=None,
    ):
        tasks = [
            tuple([c for _, c in t['subsets']])
            for t in self.config['data_schedule']
        ]
        if task_index is not None:
            tasks = [tasks[task_index]]
        k = 5

        # Overall counts
        total_overall = 0.
        correct_1_overall = 0.
        correct_k_overall = 0.
        correct_expert_overall = 0.
        correct_assign_overall = 0.

        # Loop over each task
        for task_index, task_subsets in enumerate(tasks, task_index or 0):
            # Task-wise counts
            total = 0.
            correct_1 = 0.
            correct_k = 0.
            correct_expert = 0.
            correct_assign = 0.

            # Loop over each subset
            for subset in task_subsets:
                data = DataLoader(
                    self.subsets[subset],
                    batch_size=self.config['eval_batch_size'],
                    num_workers=self.config['eval_num_workers'],
                    collate_fn=self.collate_fn,
                )
                for x, y in iter(data):
                    with torch.no_grad():
                        logits, assignments = model(
                            x, return_assignments=True)
                    total += x.size(0)
                    correct_assign += (assignments == task_index).float().sum()
                    if not self.config['disable_d']:
                        # NDPM accuracy
                        _, pred_topk = logits.topk(k, dim=1)
                        correct_topk = (
                            pred_topk.cpu()
                            == y.unsqueeze(1).expand_as(pred_topk)
                        ).float()
                        correct_1 += correct_topk[:, :1].view(-1).sum()
                        correct_k += correct_topk[:, :k].view(-1).sum()

                        # Hard-assigned expert accuracy
                        num_experts = len(model.ndpm.experts) - 1
                        if num_experts > task_index:
                            expert = model.ndpm.experts[task_index + 1]
                            with torch.no_grad():
                                logits = expert(x)
                            correct = (y == logits.argmax(dim=1).cpu()).float()
                            correct_expert += correct.sum()

            # Add to overall counts
            total_overall += total
            correct_1_overall += correct_1
            correct_k_overall += correct_k
            correct_expert_overall += correct_expert
            correct_assign_overall += correct_assign

            # Task-wise accuracies
            accuracy_1 = correct_1 / total
            accuracy_k = correct_k / total
            accuracy_expert = correct_expert / total
            accuracy_assign = correct_assign / total

            # Summarize task-wise accuracies
            writer.add_scalar(
                'accuracy_1/%s/%s/%s' % (eval_title, self.name, task_index),
                accuracy_1, step
            )
            writer.add_scalar(
                'accuracy_%s/%s/%s/%s' %
                (k, eval_title, self.name, task_index), accuracy_k, step
            )
            writer.add_scalar(
                'accuracy_expert/%s/%s/%s' %
                (eval_title, self.name, task_index), accuracy_expert, step
            )
            writer.add_scalar(
                'accuracy_assign/%s/%s/%s' %
                (eval_title, self.name, task_index), accuracy_assign, step
            )

        # Overall accuracies
        accuracy_1 = correct_1_overall / total_overall
        accuracy_k = correct_k_overall / total_overall
        accuracy_expert = correct_expert_overall / total_overall
        accuracy_assign = correct_assign_overall / total_overall

        # Summarize overall accuracies
        writer.add_scalar(
            'accuracy_1/%s/%s/overall' % (eval_title, self.name),
            accuracy_1, step
        )
        writer.add_scalar(
            'accuracy_%s/%s/%s/overall' % (k, eval_title, self.name),
            accuracy_k, step
        )
        writer.add_scalar(
            'accuracy_expert/%s/%s/overall' %
            (eval_title, self.name), accuracy_expert, step
        )
        writer.add_scalar(
            'accuracy_assign/%s/%s/overall' %
            (eval_title, self.name), accuracy_assign, step
        )

    def _eval_discriminative_model(
            self,
            model: Model,
            writer: SummaryWriter,
            step, eval_title):
        training = model.training
        model.eval()

        K = 5
        totals = []
        corrects_1 = []
        corrects_k = []

        # Accuracy of each subset
        for subset_name, subset in self.subsets.items():
            data = DataLoader(
                subset,
                batch_size=self.config['eval_batch_size'],
                num_workers=self.config['eval_num_workers'],
                collate_fn=self.collate_fn,
            )
            total = 0.
            correct_1 = 0.
            correct_k = 0.

            for x, y in iter(data):
                b = x.size(0)
                with torch.no_grad():
                    logits = model(x).view(b, -1)
                # [B, K]
                _, pred_topk = logits.topk(K, dim=1)
                correct_topk = (
                    pred_topk.cpu() == y.view(b, -1).expand_as(pred_topk)
                ).float()
                correct_1 += correct_topk[:, :1].view(-1).cpu().sum()
                correct_k += correct_topk[:, :K].view(-1).cpu().sum()
                total += x.size(0)
            totals.append(total)
            corrects_1.append(correct_1)
            corrects_k.append(correct_k)
            accuracy_1 = correct_1 / total
            accuracy_k = correct_k / total
            writer.add_scalar(
                'accuracy_1/%s/%s/%s' % (eval_title, self.name, subset_name),
                accuracy_1, step
            )
            writer.add_scalar(
                'accuracy_%d/%s/%s/%s' %
                (K, eval_title, self.name, subset_name), accuracy_k, step
            )

        # Overall accuracy
        total = sum(totals)
        correct_1 = sum(corrects_1)
        correct_k = sum(corrects_k)
        accuracy_1 = correct_1 / total
        accuracy_k = correct_k / total
        writer.add_scalar(
            'accuracy_1/%s/%s/overall' % (eval_title, self.name),
            accuracy_1, step
        )
        writer.add_scalar(
            'accuracy_%d/%s/%s/overall' % (K, eval_title, self.name),
            accuracy_k, step
        )
        model.train(training)

    def _eval_generative_model(
            self,
            model: Model,
            writer: SummaryWriter,
            step, eval_title):
        # change the model to eval mode
        training = model.training
        z_samples = model.config['z_samples']
        model.eval()
        model.config['z_samples'] = 16
        # evaluate generative model on each subset
        subset_counts = []
        subset_cumulative_bpds = []
        for subset_name, subset in self.subsets.items():
            data = DataLoader(
                subset,
                batch_size=self.config['eval_batch_size'],
                num_workers=self.config['eval_num_workers'],
                collate_fn=self.collate_fn,
            )
            subset_count = 0
            subset_cumulative_bpd = 0
            # evaluate on a subset
            for x, _ in iter(data):
                dim = reduce(lambda x, y: x * y, x.size()[1:])
                with torch.no_grad():
                    ll = model(x)
                bpd = -ll / math.log(2) / dim
                subset_count += x.size(0)
                subset_cumulative_bpd += bpd.sum()
            # append the subset evaluation result
            subset_counts.append(subset_count)
            subset_cumulative_bpds.append(subset_cumulative_bpd)
            subset_bpd = subset_cumulative_bpd / subset_count
            writer.add_scalar(
                'bpd/%s/%s/%s' % (eval_title, self.name, subset_name),
                subset_bpd, step
            )
        # Overall accuracy
        overall_bpd = sum(subset_cumulative_bpds) / sum(subset_counts)
        writer.add_scalar(
            'bpd/%s/%s/overall' % (eval_title, self.name), overall_bpd, step
        )
        # roll back the mode
        model.train(training)
        model.config['z_samples'] = z_samples

    def offset_label(self):
        if 'label_offset' not in self.config:
            return

        if isinstance(self.targets, torch.Tensor):
            self.targets += self.config['label_offset'][self.name]
        else:
            for i in range(len(self.targets)):
                self.targets[i] += self.config['label_offset'][self.name]


# =================
# Concrete Datasets
# =================

class MNIST(torchvision.datasets.MNIST, ClassificationDataset):
    name = 'mnist'
    num_classes = 10

    def __init__(self, config, train=True):
        # Compose transformation
        transform_list = [
            transforms.Resize((config['x_h'], config['x_w'])),
            transforms.ToTensor(),
        ]
        if config['recon_loss'] == 'bernoulli':
            transform_list.append(
                lambda x: (torch.rand_like(x) < x).to(torch.float)
            )
        if config['x_c'] > 1:
            transform_list.append(
                lambda x: x.expand(config['x_c'], -1, -1)
            )
        transform = transforms.Compose(transform_list)

        # Initialize super classes
        torchvision.datasets.MNIST.__init__(
            self, root=os.path.join(config['data_root'], 'mnist'),
            train=train, transform=transform, download=True)
        ClassificationDataset.__init__(self, config, train)

        # Create subset for each class
        for y in range(self.num_classes):
            self.subsets[y] = Subset(
                self,
                list((self.targets == y).nonzero().squeeze(1).numpy())
            )

        self.offset_label()


class SVHN(torchvision.datasets.SVHN, ClassificationDataset):
    name = 'svhn'
    num_classes = 10

    def __init__(self, config, train=True):
        # Compose transformation
        transform_list = [
            transforms.Resize((config['x_h'], config['x_w'])),
            transforms.ToTensor(),
        ]
        transform = transforms.Compose(transform_list)

        # Initialize super classes
        split = 'train' if train else 'test'
        torchvision.datasets.SVHN.__init__(
            self, root=os.path.join(config['data_root'], 'svhn'),
            split=split, transform=transform, download=True)
        ClassificationDataset.__init__(self, config, train)

        # Create subset for each class
        self.targets = torch.Tensor(self.labels)
        for y in range(self.num_classes):
            self.subsets[y] = Subset(
                self,
                list((self.targets == y
                      ).nonzero().squeeze(1).numpy())
            )

        self.offset_label()


class CIFAR10(torchvision.datasets.CIFAR10, ClassificationDataset):
    name = 'cifar10'
    num_classes = 10

    def __init__(self, config, train=True):
        if config.get('augment_cifar'):
            transform = transforms.Compose([
                transforms.Resize((config['x_h'], config['x_w'])),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((config['x_h'], config['x_w'])),
                transforms.ToTensor(),
            ])
        torchvision.datasets.CIFAR10.__init__(
            self, root=os.path.join(config['data_root'], 'cifar10'),
            train=train, transform=transform, download=True)
        ClassificationDataset.__init__(self, config, train)

        # Create subset for each class
        for y in range(self.num_classes):
            self.subsets[y] = Subset(
                self,
                list((torch.Tensor(self.targets) == y
                      ).nonzero().squeeze(1).numpy())
            )

        self.offset_label()


class CIFAR100(torchvision.datasets.CIFAR100, ClassificationDataset):
    name = 'cifar100'
    num_classes = 100

    def __init__(self, config, train=True):
        if config.get('augment_cifar'):
            transform = transforms.Compose([
                transforms.Resize((config['x_h'], config['x_w'])),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((config['x_h'], config['x_w'])),
                transforms.ToTensor(),
            ])
        torchvision.datasets.CIFAR100.__init__(
            self, root=os.path.join(config['data_root'], 'cifar100'),
            train=train, transform=transform, download=True)
        ClassificationDataset.__init__(self, config, train)

        # Create subset for each class
        for y in range(self.num_classes):
            self.subsets[y] = Subset(
                self,
                list((torch.Tensor(self.targets) == y
                      ).nonzero().squeeze(1).numpy())
            )

        self.offset_label()


DATASET = {
    MNIST.name: MNIST,
    SVHN.name: SVHN,
    CIFAR10.name: CIFAR10,
    CIFAR100.name: CIFAR100,
}
