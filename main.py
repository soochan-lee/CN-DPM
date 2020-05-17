#!/usr/bin/env python3
from argparse import ArgumentParser
import os
import yaml
import resource
import torch
from tensorboardX import SummaryWriter
from data import DataScheduler
from models import MODEL
from train import train_model


# Increase maximum number of open files from 1024 to 4096
# as suggested in https://github.com/pytorch/pytorch/issues/973
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

parser = ArgumentParser()
parser.add_argument(
    '--config', '-c', default='configs/mlp_classifier-mlp_vae-split_mnist.yaml'
)
parser.add_argument(
    '--episode', '-e', default='episodes/mnist-split-online.yaml'
)
parser.add_argument('--log-dir', '-l')
parser.add_argument('--override', default='')


def main():
    args = parser.parse_args()

    # Load config
    config = yaml.load(open(args.config), Loader=yaml.FullLoader)
    episode = yaml.load(open(args.episode), Loader=yaml.FullLoader)
    config['data_schedule'] = episode

    # Override options
    for option in args.override.split('|'):
        if not option:
            continue
        address, value = option.split('=')
        keys = address.split('.')
        here = config
        for key in keys[:-1]:
            if key not in here:
                raise ValueError('{} is not defined in config file. '
                                 'Failed to override.'.format(address))
            here = here[key]
        if keys[-1] not in here:
            raise ValueError('{} is not defined in config file. '
                             'Failed to override.'.format(address))
        here[keys[-1]] = yaml.load(value, Loader=yaml.FullLoader)

    # Set log directory
    config['log_dir'] = args.log_dir
    if os.path.exists(args.log_dir):
        print('WARNING: %s already exists' % args.log_dir)
        input('Press enter to continue')

    # Save config
    os.makedirs(config['log_dir'], mode=0o755, exist_ok=True)
    config_save_path = os.path.join(config['log_dir'], 'config.yaml')
    episode_save_path = os.path.join(config['log_dir'], 'episode.yaml')
    yaml.dump(config, open(config_save_path, 'w'))
    yaml.dump(episode, open(episode_save_path, 'w'))
    print('Config & episode saved to {}'.format(config['log_dir']))

    # Build components
    data_scheduler = DataScheduler(config)
    writer = SummaryWriter(config['log_dir'])
    model = MODEL[config['model_name']](config, writer)
    model.to(config['device'])
    train_model(config, model, data_scheduler, writer)


if __name__ == '__main__':
    main()
