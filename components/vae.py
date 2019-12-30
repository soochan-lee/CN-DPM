from abc import ABC, abstractmethod
from itertools import accumulate
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List, Tuple

from ndpm.summaries import VaeSummaries
from loss import bernoulli_nll, logistic_nll, gaussian_nll, laplace_nll
from torch.nn.functional import relu
from .component import ComponentG
from .classifier import conv4x4t, conv1x1, BasicBlock
from utils import Lambda


class Vae(ComponentG, ABC):
    def __init__(self, config, experts):
        super().__init__(config, experts)
        self.x_c = config['x_c']
        bernoulli = config['recon_loss'] == 'bernoulli'
        if bernoulli:
            self.log_var_param = None
        elif config['learn_x_log_var']:
            self.log_var_param = nn.Parameter(
                torch.ones([self.x_c]) * config['x_log_var_param'],
                requires_grad=True
            )
        else:
            self.log_var_param = (
                torch.ones([self.x_c], device=self.device) *
                config['x_log_var_param']
            )

    def forward(self, x):
        x = x.to(self.device)
        z_mean, z_log_var = self.encode(x)
        z = self.reparameterize(z_mean, z_log_var, 1)
        return self.decode(z)

    def nll(self, x, y=None, step=None) -> Tuple[Tensor, VaeSummaries]:
        x = x.to(self.device)
        z_mean, z_log_var = self.encode(x)
        z = self.reparameterize(z_mean, z_log_var, self.config['z_samples'])
        x_mean = self.decode(z)
        x_mean = x_mean.view(x.size(0), self.config['z_samples'], *x.shape[1:])
        x_log_var = (
            None if self.config['recon_loss'] == 'bernoulli' else
            self.log_var.view(1, 1, -1, 1, 1)
        )
        loss_recon = self.reconstruction_loss(x, x_mean, x_log_var)
        loss_recon = loss_recon.view(x.size(0), self.config['z_samples'], -1)
        loss_recon = loss_recon.sum(2).mean(1)
        loss_kl = self.gaussian_kl(z_mean, z_log_var)
        loss_vae = loss_recon + loss_kl

        summaries = VaeSummaries(
            loss_recon=loss_recon,
            loss_kl=loss_kl,
            loss_vae=loss_vae,
            z_mean=z_mean,
            z_log_var=z_log_var,
            x=x,
            x_mean=x_mean,
        )
        return loss_vae, summaries

    def sample(self, n=1):
        z = torch.randn(n, self.config['z_dim'], device=self.device)
        x_mean = self.decode(z)
        return x_mean

    def reconstruction_loss(self, x, x_mean, x_log_var=None):
        loss_type = self.config['recon_loss']
        loss = (
            bernoulli_nll if loss_type == 'bernoulli' else
            gaussian_nll if loss_type == 'gaussian' else
            laplace_nll if loss_type == 'laplace' else
            logistic_nll if loss_type == 'logistic' else None
        )
        if loss is None:
            raise ValueError('Unknown recon_loss type: {}'.format(loss_type))

        if len(x_mean.size()) > len(x.size()):
            x = x.unsqueeze(1)

        return (
            loss(x, x_mean) if x_log_var is None else
            loss(x, x_mean, x_log_var)
        )

    @staticmethod
    def gaussian_kl(q_mean, q_log_var, p_mean=None, p_log_var=None):
        # p defaults to N(0, 1)
        zeros = torch.zeros_like(q_mean)
        p_mean = p_mean if p_mean is not None else zeros
        p_log_var = p_log_var if p_log_var is not None else zeros
        # calcaulate KL(q, p)
        kld = 0.5 * (
            p_log_var - q_log_var +
            (q_log_var.exp() + (q_mean - p_mean) ** 2) / p_log_var.exp() - 1
        )
        kld = kld.sum(1)
        return kld

    @staticmethod
    def reparameterize(z_mean, z_log_var, num_samples=1):
        z_std = (z_log_var * 0.5).exp()
        z_std = z_std.unsqueeze(1).expand(-1, num_samples, -1)
        z_mean = z_mean.unsqueeze(1).expand(-1, num_samples, -1)
        unit_normal = torch.randn_like(z_std)
        z = z_mean + unit_normal * z_std
        z = z.view(-1, z_std.size(2))
        return z

    @abstractmethod
    def encode(self, x):
        pass

    @abstractmethod
    def decode(self, x):
        pass

    @property
    def log_var(self):
        return (
            None if self.log_var_param is None else
            self.log_var_param
        )


class SharingVae(Vae, ABC):
    def collect_nll(self, x, y=None, step=None) \
            -> Tuple[Tensor, List[VaeSummaries]]:
        """Collect NLL values

        Returns:
            loss_vae: Tensor of shape [B, 1+K]
            summaries: VAE summaries
        """
        x = x.to(self.device)

        # Dummy VAE
        dummy_nll, dummy_summary = self.experts[0].g.nll(x, y, step)

        # Encode
        z_means, z_log_vars, features = self.encode(x, collect=True)

        # Decode
        loss_vaes, summaries = [dummy_nll], [dummy_summary]
        vaes = [expert.g for expert in self.experts[1:]] + [self]
        x_logits = []
        for z_mean, z_log_var, vae in zip(z_means, z_log_vars, vaes):
            z = self.reparameterize(z_mean, z_log_var, self.config['z_samples'])
            if self.config.get('precursor_conditioned_decoder'):
                x_logit = vae.decode(z, as_logit=True)
                x_logits.append(x_logit)
                continue
            x_mean = vae.decode(z)
            x_mean = x_mean.view(x.size(0), self.config['z_samples'],
                                 *x.shape[1:])
            x_log_var = (
                None if self.config['recon_loss'] == 'bernoulli' else
                self.log_var.view(1, 1, -1, 1, 1)
            )
            loss_recon = self.reconstruction_loss(x, x_mean, x_log_var)
            loss_recon = loss_recon.view(x.size(0), self.config['z_samples'],
                                         -1)
            loss_recon = loss_recon.sum(2).mean(1)
            loss_kl = self.gaussian_kl(z_mean, z_log_var)
            loss_vae = loss_recon + loss_kl

            loss_vaes.append(loss_vae)
            summaries.append(VaeSummaries(
                loss_recon=loss_recon,
                loss_kl=loss_kl,
                loss_vae=loss_vae,
                z_mean=z_mean,
                z_log_var=z_log_var,
                x=x,
                x_mean=x_mean,
            ))

        x_logits = list(accumulate(
            x_logits, func=(lambda x, y: x.detach() + y)
        ))
        for x_logit in x_logits:
            x_mean = torch.sigmoid(x_logit)
            x_mean = x_mean.view(x.size(0), self.config['z_samples'],
                                 *x.shape[1:])
            x_log_var = (
                None if self.config['recon_loss'] == 'bernoulli' else
                self.log_var.view(1, 1, -1, 1, 1)
            )
            loss_recon = self.reconstruction_loss(x, x_mean, x_log_var)
            loss_recon = loss_recon.view(x.size(0), self.config['z_samples'],
                                         -1)
            loss_recon = loss_recon.sum(2).mean(1)
            loss_kl = self.gaussian_kl(z_mean, z_log_var)
            loss_vae = loss_recon + loss_kl
            summary_x_log_var = (
                [[[[None], [None], [None]]]] if x_log_var is None else
                x_log_var
            )
            loss_vaes.append(loss_vae)
            summaries.append(VaeSummaries(
                loss_recon=loss_recon,
                loss_kl=loss_kl,
                loss_vae=loss_vae,
                z_mean=z_mean,
                z_log_var=z_log_var,
                x=x,
                x_mean=x_mean,
                x_log_var_r=summary_x_log_var[0][0][0][0],
                x_log_var_g=summary_x_log_var[0][0][1][0],
                x_log_var_b=summary_x_log_var[0][0][2][0],
            ))

        return torch.stack(loss_vaes, dim=1), summaries

    @abstractmethod
    def encode(self, x, collect=False):
        pass

    @abstractmethod
    def decode(self, z, as_logit=False):
        """
        Decode do not share parameters
        """
        pass


class CnnVae(Vae):
    def __init__(self, config, experts):
        super().__init__(config, experts)
        nf = config['vae_nf']
        h1_dim = 1 * nf
        h2_dim = 2 * nf
        fc_dim = 4 * nf

        feature_volume = (
                (config['x_h'] // 4) *
                (config['x_w'] // 4) *
                h2_dim)
        self.encoder = nn.Sequential(
            nn.Conv2d(config['x_c'], h1_dim,
                      kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(h1_dim, h2_dim,
                      kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2),
            nn.ReLU(),
            Lambda(lambda x: x.view(x.size(0), -1)),
            nn.Linear(feature_volume, fc_dim)
        )
        self.enc_fc_z_mean = nn.Linear(fc_dim, config['z_dim'])
        self.enc_fc_z_log_var = nn.Linear(fc_dim, config['z_dim'])

        self.decoder = nn.Sequential(
            nn.Linear(config['z_dim'], fc_dim),
            nn.Linear(fc_dim, feature_volume),
            Lambda(lambda x: x.view(
                x.size(0), h2_dim,
                config['x_h'] // 4, config['x_w'] // 4)),
            nn.ReLU(),
            nn.ConvTranspose2d(h2_dim, h1_dim,
                               kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(h1_dim, config['x_c'],
                               kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

        self.to(self.device)
        self.setup_optimizer()

    def encode(self, x):
        h = self.encoder(x)
        z_mean = self.enc_fc_z_mean(h)
        z_log_var = self.enc_fc_z_log_var(h)
        return z_mean, z_log_var

    def decode(self, z):
        return self.decoder(z)


class CnnSharingVae(SharingVae):
    def __init__(self, config, experts):
        super().__init__(config, experts)
        self.precursors = [expert.g for expert in self.experts[1:]]
        first = len(self.precursors) == 0
        nf_base, nf_ext = config['vae_nf_base'], config['vae_nf_ext']
        nf = nf_base if first else nf_ext
        nf_cat = nf_base + len(self.precursors) * nf_ext

        h1_dim = 1 * nf
        h2_dim = 2 * nf
        fc_dim = 4 * nf
        h1_cat_dim = 1 * nf_cat
        h2_cat_dim = 2 * nf_cat
        fc_cat_dim = 4 * nf_cat

        self.fc_dim = fc_dim
        feature_volume = ((config['x_h'] // 4) * (config['x_w'] // 4) *
                          h2_cat_dim)

        self.enc1 = nn.Sequential(
            nn.Conv2d(config['x_c'], h1_dim, 3, 1, 1),
            nn.MaxPool2d(2),
            nn.ReLU()
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(h1_cat_dim, h2_dim, 3, 1, 1),
            nn.MaxPool2d(2),
            nn.ReLU(),
            Lambda(lambda x: x.view(x.size(0), -1))
        )
        self.enc3 = nn.Sequential(
            nn.Linear(feature_volume, fc_dim),
            nn.ReLU()
        )
        self.enc_z_mean = nn.Linear(fc_cat_dim, config['z_dim'])
        self.enc_z_log_var = nn.Linear(fc_cat_dim, config['z_dim'])

        self.dec_z = nn.Sequential(
            nn.Linear(config['z_dim'], 4 * nf_base),
            nn.ReLU()
        )
        self.dec3 = nn.Sequential(
            nn.Linear(
                4 * nf_base,
                (config['x_h'] // 4) * (config['x_w'] // 4) * 2 * nf_base),
            nn.ReLU()
        )
        self.dec2 = nn.Sequential(
            Lambda(lambda x: x.view(
                x.size(0), 2 * nf_base,
                config['x_h'] // 4, config['x_w'] // 4)),
            nn.ConvTranspose2d(2 * nf_base, 1 * nf_base,
                               kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )
        self.dec1 = nn.ConvTranspose2d(1 * nf_base, config['x_c'],
                                       kernel_size=4, stride=2, padding=1)

        self.to(self.device)
        self.setup_optimizer()

    def encode(self, x, collect=False):
        # When first component
        if len(self.precursors) == 0:
            h1 = self.enc1(x)
            h2 = self.enc2(h1)
            h3 = self.enc3(h2)
            z_mean = self.enc_z_mean(h3)
            z_log_var = self.enc_z_log_var(h3)

            if collect:
                return [z_mean], [z_log_var], \
                       [h1.detach(), h2.detach(), h3.detach()]
            else:
                return z_mean, z_log_var

        # Second or later component
        z_means, z_log_vars, features = \
            self.precursors[-1].encode(x, collect=True)

        h1 = self.enc1(x)
        h1_cat = torch.cat([features[0], h1], dim=1)
        h2 = self.enc2(h1_cat)
        h2_cat = torch.cat([features[1], h2], dim=1)
        h3 = self.enc3(h2_cat)
        h3_cat = torch.cat([features[2], h3], dim=1)
        z_mean = self.enc_z_mean(h3_cat)
        z_log_var = self.enc_z_log_var(h3_cat)

        if collect:
            z_means.append(z_mean)
            z_log_vars.append(z_log_var)
            features = [h1_cat.detach(), h2_cat.detach(), h3_cat.detach()]
            return z_means, z_log_vars, features
        else:
            return z_mean, z_log_var

    def decode(self, z, as_logit=False):
        h3 = self.dec_z(z)
        h2 = self.dec3(h3)
        h1 = self.dec2(h2)
        x_logit = self.dec1(h1)
        return x_logit if as_logit else torch.sigmoid(x_logit)


class MlpVae(Vae):
    def __init__(self, config, experts):
        super().__init__(config, experts)
        vae_nf = config['vae_nf']
        self.enc_fc1 = nn.Linear(config['x_c'] * config['x_h'] * config['x_w'],
                                 vae_nf)
        self.enc_fc2 = nn.Linear(vae_nf, vae_nf)
        self.enc_fc_z_mean = nn.Linear(vae_nf, config['z_dim'])
        self.enc_fc_z_log_var = nn.Linear(vae_nf, config['z_dim'])
        self.dec_fc1 = nn.Linear(config['z_dim'], vae_nf)
        self.dec_fc2 = nn.Linear(vae_nf, vae_nf)
        self.dec_fc_x_mean = nn.Linear(
            vae_nf, config['x_c'] * config['x_h'] * config['x_w'])

        self.to(self.device)
        self.setup_optimizer()

    def encode(self, x):
        x = x.view(x.size(0), -1)
        h = F.relu(self.enc_fc1(x))
        h = F.relu(self.enc_fc2(h))
        z_mean = self.enc_fc_z_mean(h)
        z_log_var = self.enc_fc_z_log_var(h)
        return z_mean, z_log_var

    def decode(self, z):
        h = F.relu(self.dec_fc1(z))
        h = F.relu(self.dec_fc2(h))
        x_mean = torch.sigmoid(self.dec_fc_x_mean(h)).view(
            z.size(0), self.config['x_c'],
            self.config['x_h'], self.config['x_w']
        )
        return x_mean


class MlpSharingVae(SharingVae):
    def __init__(self, config, experts):
        super().__init__(config, experts)
        self.precursors = [expert.g for expert in self.experts[1:]]
        first = len(self.precursors) == 0
        vae_nf_base = config['vae_nf_base']
        vae_nf_ext = config['vae_nf_ext']
        h1_dim = vae_nf_base if first else vae_nf_ext
        h2_dim = vae_nf_base if first else vae_nf_ext
        h1_cat_dim = vae_nf_base \
                     + len(self.precursors) * vae_nf_ext
        h2_cat_dim = vae_nf_base \
                     + len(self.precursors) * vae_nf_ext
        self.h2_dim = h2_dim
        self.enc1 = nn.Sequential(
            nn.Linear(config['x_c'] * config['x_h'] * config['x_w'], h1_dim),
            nn.ReLU()
        )
        self.enc2 = nn.Sequential(
            nn.Linear(h1_cat_dim, h2_dim),
            nn.ReLU()
        )
        self.enc_z_mean = nn.Linear(h2_cat_dim, config['z_dim'])
        self.enc_z_log_var = nn.Linear(h2_cat_dim, config['z_dim'])
        self.dec_z = nn.Sequential(
            nn.Linear(config['z_dim'], vae_nf_base),
            nn.ReLU()
        )
        self.dec2 = nn.Sequential(
            nn.Linear(vae_nf_base, vae_nf_base),
            nn.ReLU()
        )
        self.dec1 = nn.Linear(
            vae_nf_base, config['x_c'] * config['x_h'] * config['x_w']
        )

        self.to(self.device)
        self.setup_optimizer()

    def encode(self, x, collect=False):
        # When first component
        x_flat = x.to(self.device).view(x.size(0), -1)
        if len(self.precursors) == 0:
            h1 = self.enc1(x_flat)
            h2 = self.enc2(h1)
            z_mean = self.enc_z_mean(h2)
            z_log_var = self.enc_z_log_var(h2)
            if collect:
                return [z_mean], [z_log_var], [h1.detach(), h2.detach()]
            else:
                return z_mean, z_log_var

        # Second or later component
        z_means, z_log_vars, features = \
            self.precursors[-1].encode(x, collect=True)
        h1 = self.enc1(x_flat)
        h1_cat = torch.cat([features[0], h1], dim=1)
        h2 = self.enc2(h1_cat)
        h2_cat = torch.cat([features[1], h2], dim=1)
        z_mean = self.enc_z_mean(h2_cat)
        z_log_var = self.enc_z_log_var(h2_cat)

        if collect:
            z_means.append(z_mean)
            z_log_vars.append(z_log_var)
            features = [h1_cat.detach(), h2_cat.detach()]
            return z_means, z_log_vars, features
        else:
            return z_mean, z_log_var

    def decode(self, z, as_logit=False):
        h2 = self.dec_z(z)
        h1 = self.dec2(h2)
        x_logit = self.dec1(h1)
        return x_logit if as_logit else torch.sigmoid(x_logit)


# =========================
# ResNetVae Implementations
# =========================

class ResNetEncoder(nn.Module):
    block = BasicBlock
    groups = 1
    dilation = 1
    num_blocks = [2, 2, 2, 2]
    norm_layer = nn.BatchNorm2d

    def __init__(self, config):
        super().__init__()
        num_blocks = self.num_blocks
        block = self.block
        self.device = config['device']
        self.inplanes = nf = config['vae_nf']
        self.conv1 = nn.Conv2d(
            3, nf * 1, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(nf * 1)
        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)
        self.layer5 = nn.Sequential(
            nn.Conv2d(nf * 8, nf * 8, 3, 2, 1),
            nn.ReLU()
        )

        fc_size = config['x_h']
        for i in range(4):
            fc_size = fc_size // 2 + (fc_size % 2)

        self.fc_z_mean = nn.Linear((fc_size ** 2) * nf * 8, config['z_dim'])
        self.fc_z_log_var = nn.Linear((fc_size ** 2) * nf * 8, config['z_dim'])

    def _make_layer(self, block, planes, num_blocks, stride, dilate=False):
        norm_layer = self.norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
        layers = []
        layers.append(block(
            self.inplanes, planes, stride, downsample,
            groups=self.groups,
            dilation=previous_dilation,
            norm_layer=norm_layer
        ))
        self.inplanes = planes * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(
                self.inplanes, planes, groups=self.groups,
                dilation=self.dilation, norm_layer=norm_layer
            ))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.to(self.device)
        out = relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = out.view(out.size(0), -1)
        z_mean = self.fc_z_mean(out)
        z_log_var = self.fc_z_log_var(out)
        return z_mean, z_log_var


class ResNetDecoder(nn.Module):
    block = BasicBlock
    groups = 1
    dilation = 1
    num_blocks = [2, 2, 2, 2]
    norm_layer = nn.BatchNorm2d

    def __init__(self, config):
        super().__init__()
        num_blocks = self.num_blocks
        block = self.block
        z_dim = config['z_dim']
        nf = config['vae_nf']

        f_sizes = [config['x_h']]
        for i in range(4):
            f_sizes.append(f_sizes[-1] // 2 + (f_sizes[-1] % 2))
        self.f_sizes = f_sizes

        self.device = config['device']
        self.inplanes = nf * 8
        self.projection = nn.Linear(
            z_dim,
            self.inplanes * (self.f_sizes[-1] ** 2),
            bias=False,
        )
        self.layer5 = nn.Sequential(
            nn.ConvTranspose2d(nf * 8, nf * 8, 4, 2, 1),
            nn.ReLU()
        )
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        self.bn1 = nn.BatchNorm2d(nf * 1)
        self.conv1 = nn.Conv2d(
           nf * 1, 3, kernel_size=3, stride=1, padding=1, bias=False
        )

    def forward(self, z):
        b = z.size(0)
        z = z.to(self.device)
        out = self.projection(z)
        out = out.view(b, -1, self.f_sizes[-1], self.f_sizes[-1])
        out = self.layer5(out)
        out = out[:, :, :self.f_sizes[-2], :self.f_sizes[-2]]
        out = self.layer4(out)
        out = out[:, :, :self.f_sizes[-3], :self.f_sizes[-3]]
        out = self.layer3(out)
        out = out[:, :, :self.f_sizes[-4], :self.f_sizes[-4]]
        out = self.layer2(out)
        out = self.layer1(out)
        out = self.bn1(out)
        return torch.sigmoid(self.conv1(out))

    def _make_layer(self, block, planes, num_blocks, stride):
        norm_layer = self.norm_layer
        upsample = None
        previous_dilation = self.dilation
        if stride != 1:
            upsample = nn.Sequential(
                conv4x4t(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
        elif self.inplanes != planes * block.expansion:
            upsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(
            self.inplanes, planes, stride, upsample=upsample,
            groups=self.groups,
            dilation=previous_dilation,
            norm_layer=norm_layer
        ))
        self.inplanes = planes * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(
                self.inplanes, planes, groups=self.groups,
                dilation=self.dilation, norm_layer=norm_layer
            ))

        return nn.Sequential(*layers)


class ResNetVae(Vae):
    def __init__(self, config, experts):
        super().__init__(config, experts)
        self.encoder = ResNetEncoder(config)
        self.decoder = ResNetDecoder(config)
        self.to(self.device)
        self.setup_optimizer()

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)
