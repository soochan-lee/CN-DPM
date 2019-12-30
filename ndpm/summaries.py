import torch


class Summaries(object):
    scalars = None
    images = None
    histograms = None
    optionals = None

    def __init__(self, tensor_writes=None,
                 _scalars=None, _histograms=None,
                 _images=None, _optionals=None, **tensors):
        # initialize instance field attributes (defaults to the cls attributes)
        self._scalars = _scalars or self.scalars or {}
        self._histograms = _histograms or self.histograms or {}
        self._images = _images or self.images or {}
        self._optionals = list(_optionals or self.optionals or [])
        # validate tensors with the instance requirement attributes

        required_tensors = ((
            set([n for n in self._scalars.values() if isinstance(n, str)]) |
            set([n for n in self._images.values() if isinstance(n, str)]) |
            set([n for n in self._histograms.values() if isinstance(n, str)])
        ) - set(self._optionals))
        for name in required_tensors:
            if name not in tensors:
                raise ValueError((
                    'Required tensor {} is missing in the summary argument. '
                    'Expected {}, but given {}.'
                ).format(name, required_tensors, tensors.keys()))
        # set main instance attributes
        self.tensors = tensors
        self.tensor_writes = tensor_writes or [
            getattr(self, name) for name in dir(self) if
            callable(getattr(self, name)) and name.startswith('_write_')
        ]

    def write(self, writer, step, prefix='', postfix=''):
        for summary_name, maybe_tensor in self._scalars.items():
            if self._should_skip(summary_name, maybe_tensor):
                continue
            tensor = (
                maybe_tensor if isinstance(maybe_tensor, torch.Tensor) else
                self.tensors[maybe_tensor]
            )

            writer.add_scalar(
                '{}{}{}'.format(prefix, summary_name, postfix),
                tensor, step
            )
        # write histograms
        for summary_name, maybe_tensor in self._histograms.items():
            if self._should_skip(summary_name, maybe_tensor):
                continue
            tensor = (
                maybe_tensor if isinstance(maybe_tensor, torch.Tensor) else
                self.tensors[maybe_tensor]
            )

            writer.add_histogram(
                '{}{}{}'.format(prefix, summary_name, postfix),
                tensor, step
            )

        # write images
        for summary_name, maybe_tensor in self._images.items():
            if self._should_skip(summary_name, maybe_tensor):
                continue
            tensor = (
                maybe_tensor if isinstance(maybe_tensor, torch.Tensor) else
                self.tensors[maybe_tensor]
            )
            writer.add_image(
                '{}{}{}'.format(prefix, summary_name, postfix),
                tensor, step
            )
        # run write-methods
        for tensor_write in self.tensor_writes:
            tensor_write(writer, step, prefix=prefix, postfix=postfix)

    def add_tensor_summary(self, summary_name, tensor, type):
        tensor = tensor.detach()
        assert type in ('scalar', 'image', 'histogram')
        if type == 'scalar':
            self._scalars[summary_name] = tensor
        elif type == 'image':
            self._images[summary_name] = tensor
        else:
            self._histograms[summary_name] = tensor

    def __add__(self, other):
        scalars = {}
        scalars.update(self._scalars)
        scalars.update(other._scalars)
        images = {}
        images.update(self._images)
        images.update(other._images)
        histograms = {}
        histograms.update(self._histograms)
        histograms.update(other._histograms)
        tensors = {}
        tensors.update(self.tensors)
        tensors.update(other.tensors)
        return Summaries(
            tensor_writes=(self.tensor_writes + other.tensor_writes),
            _scalars=scalars, _images=images, _histograms=histograms,
            _optionals=(self._optionals + other._optionals),
            **tensors
        )

    def _should_skip(self, summary_name, maybe_tensor):
        return (
            isinstance(maybe_tensor, str) and
            (maybe_tensor not in self.tensors or self.tensors[maybe_tensor] is None) and
            summary_name in self._optionals
        )


class VaeSummaries(Summaries):
    scalars = {
        'x_log_var_r': 'x_log_var_r',
        'x_log_var_g': 'x_log_var_g',
        'x_log_var_b': 'x_log_var_b',
    }
    histograms = {
        'z_mean': 'z_mean',
        'z_log_var': 'z_log_var',
        'loss_hist/recon': 'loss_recon',
        'loss_hist/kl': 'loss_kl',
        'loss_hist/vae': 'loss_vae',
    }
    optionals = (
        'z_mean',
        'z_log_var',
        'x_log_var_r',
        'x_log_var_g',
        'x_log_var_b',
    )

    def _write_kld_per_z(self, writer, step, prefix='', postfix=''):
        if 'z_mean' not in self.tensors or 'z_log_var' not in self.tensors:
            return
        # calculate kld z_log_var = self.tensors['z_log_var']
        z_mean = self.tensors['z_mean']
        z_log_var = self.tensors['z_log_var']
        kl_loss = 0.5 * (z_log_var.exp() + z_mean ** 2 - 1 - z_log_var)
        # and write kld per z histogram
        writer.add_histogram(
            '{}kld_per_z{}'.format(prefix, postfix), kl_loss.mean(0), step
        )

    def _write_recon_collage(self, writer, step, prefix='', postfix=''):
        if 'x_mean' not in self.tensors or 'x' not in self.tensors:
            return
        x = self.tensors['x']
        x_mean = self.tensors['x_mean']
        # build an collage
        input_images = []
        output_images = []
        for i in range(min(x.size(0), 4)):
            input_images.append(x[i])
            output_images.append(x_mean[i][0].view_as(x[i]))
        input_images = torch.cat(input_images, dim=2)
        output_images = torch.cat(output_images, dim=2)
        recon_collage = torch.cat([input_images, output_images], dim=1)
        # and write an image
        writer.add_image(
            '{}recon{}'.format(prefix, postfix), recon_collage.detach(), step
        )


class ClassificationSummaries(Summaries):
    scalars = {}
    histograms = {
        'loss_hist/pred': 'loss_pred'
    }
