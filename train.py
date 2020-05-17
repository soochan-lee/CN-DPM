import os
import pickle
import torch
from tensorboardX import SummaryWriter
from models import NdpmModel
from data import DataScheduler


def _write_summary(summary, writer: SummaryWriter, step):
    for summary_type, summary_dict in summary.items():
        if summary_type == 'scalar':
            write_fn = writer.add_scalar
        elif summary_type == 'image':
            write_fn = writer.add_image
        elif summary_type == 'histogram':
            write_fn = writer.add_histogram
        else:
            raise RuntimeError('Unsupported summary type: %s' % summary_type)

        for tag, value in summary_dict.items():
            write_fn(tag, value, step)


def _make_collage(samples, config, grid_h, grid_w):
    s = samples.view(
        grid_h, grid_w,
        config['x_c'], config['x_h'], config['x_w']
    )
    collage = s.permute(2, 0, 3, 1, 4).contiguous().view(
        config['x_c'],
        config['x_h'] * grid_h,
        config['x_w'] * grid_w
    )
    return collage


def train_model(config, model: NdpmModel,
                scheduler: DataScheduler,
                writer: SummaryWriter):
    for step, (x, y, t) in enumerate(scheduler):
        step += 1
        if isinstance(model, NdpmModel):
            print('\r[Step {:4}] STM: {:5}/{} | #Expert: {}'.format(
                step,
                len(model.ndpm.stm_x), config['stm_capacity'],
                len(model.ndpm.experts) - 1
            ), end='')
        else:
            print('\r[Step {:4}]'.format(step), end='')

        summarize = step % config['summary_step'] == 0
        summarize_experts = summarize and isinstance(model, NdpmModel)
        summarize_samples = summarize and config['summarize_samples']

        # learn the model
        model.learn(x, y, t, step)

        # Evaluate the model
        evaluatable = (
            not isinstance(model, NdpmModel) or len(model.ndpm.experts) > 1
        )
        if evaluatable and step % config['eval_step'] == 0:
            scheduler.eval(model, writer, step, 'model')

        # Evaluate experts of the model's DPMoE
        if summarize_experts:
            writer.add_scalar('num_experts', len(model.ndpm.experts) - 1, step)

        # Summarize samples
        if summarize_samples:
            is_ndpm = isinstance(model, NdpmModel)
            comps = [e.g for e in model.ndpm.experts[1:]] \
                if is_ndpm else [model.component]
            if len(comps) == 0:
                continue
            grid_h, grid_w = config['sample_grid']
            total_samples = []
            # Sample from each expert
            for i, expert in enumerate(comps):
                with torch.no_grad():
                    samples = expert.sample(grid_h * grid_w)
                total_samples.append(samples)
                collage = _make_collage(samples, config, grid_h, grid_w)
                writer.add_image('samples/{}'.format(i + 1), collage, step)

            if is_ndpm:
                counts = model.ndpm.prior.counts[1:]
                expert_w = counts / counts.sum()
                num_samples = torch.distributions.multinomial.Multinomial(
                    grid_h * grid_w, probs=expert_w).sample().type(torch.int)
                to_collage = []
                for i, samples in enumerate(total_samples):
                    to_collage.append(samples[:num_samples[i]])
                to_collage = torch.cat(to_collage, dim=0)
                collage = _make_collage(to_collage, config, grid_h, grid_w)
                writer.add_image('samples/ndpm', collage, step)
