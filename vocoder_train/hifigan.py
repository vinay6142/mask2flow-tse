"""
HiFi-GAN vocoder — mel spectrogram -> waveform.

Configured specifically for THIS project's mel spec (configs/default_v2.yaml):
  sample_rate=16000, n_mels=80, n_fft=1024, hop_length=160, win_length=400.

This is NOT the standard HiFi-GAN config (which targets hop=256 at 22.05/24kHz
for TTS). Upsample rates here multiply to 160 (the actual hop length) instead
of 256, so the generator's total temporal upsampling exactly matches how the
project's MelSpectrogramExtractor downsamples audio into mel frames.

Uses the classic torch.nn.utils.weight_norm (not the newer parametrize-based
torch.nn.utils.parametrizations.weight_norm) deliberately — this project
already hit a real bug from a parametrize-based weight_norm naming mismatch
(WavLM's pos_conv_embed "newly initialized" warning); the classic API keeps
state_dict keys simple and avoids that whole class of issue here.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn.utils import weight_norm, remove_weight_norm

LRELU_SLOPE = 0.1


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if "Conv" in classname:
        m.weight.data.normal_(mean, std)


# ─────────────────────────────────────────────────────────────
# Generator
# ─────────────────────────────────────────────────────────────
class ResBlock1(nn.Module):
    """Multi-receptive-field fusion block, 3 dilated conv pairs per resolution."""

    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super().__init__()
        self.convs1 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=d,
                                padding=get_padding(kernel_size, d)))
            for d in dilation
        ])
        self.convs2 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                                padding=get_padding(kernel_size, 1)))
            for _ in dilation
        ])
        self.convs1.apply(init_weights)
        self.convs2.apply(init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for c in self.convs1:
            remove_weight_norm(c)
        for c in self.convs2:
            remove_weight_norm(c)


class Generator(nn.Module):
    """
    Mel (B, n_mels, T) -> waveform (B, 1, T * hop_length).

    Default config: upsample_rates=[8,5,2,2] -> product 160 = this
    project's hop_length. upsample_initial_channel=256 (lighter than the
    original paper's 512) to keep training time reasonable on P100 GPUs;
    raise it if quality needs it and compute allows.
    """

    def __init__(
        self,
        n_mels=80,
        upsample_rates=(8, 5, 2, 2),
        upsample_kernel_sizes=(16, 10, 4, 4),
        upsample_initial_channel=256,
        resblock_kernel_sizes=(3, 7, 11),
        resblock_dilation_sizes=((1, 3, 5), (1, 3, 5), (1, 3, 5)),
    ):
        super().__init__()
        assert len(upsample_rates) == len(upsample_kernel_sizes)
        total_upsample = 1
        for r in upsample_rates:
            total_upsample *= r

        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.total_upsample = total_upsample

        self.conv_pre = weight_norm(Conv1d(n_mels, upsample_initial_channel, 7, 1, padding=3))

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            in_ch = upsample_initial_channel // (2 ** i)
            out_ch = upsample_initial_channel // (2 ** (i + 1))
            # Choose padding + output_padding so output length is EXACTLY
            # input_length * u for every (u, k) pair, including ones where
            # (k - u) is odd (e.g. k=10, u=5) where floor((k-u)/2) alone
            # would under-pad and leave the output a few samples too long.
            pad = -(-(k - u) // 2)  # ceil((k-u)/2) via integer math
            out_pad = u + 2 * pad - k
            assert 0 <= out_pad < u, f"invalid output_padding {out_pad} for k={k}, u={u}"
            self.ups.append(weight_norm(ConvTranspose1d(
                in_ch, out_ch, k, u, padding=pad, output_padding=out_pad
            )))

        self.resblocks = nn.ModuleList()
        for i in range(self.num_upsamples):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks.append(ResBlock1(ch, k, d))

        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(self, mel):
        x = self.conv_pre(mel)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                rb_out = self.resblocks[i * self.num_kernels + j](x)
                xs = rb_out if xs is None else xs + rb_out
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)
        return x  # (B, 1, T * total_upsample)

    def remove_weight_norm(self):
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)


# ─────────────────────────────────────────────────────────────
# Multi-Period Discriminator
# ─────────────────────────────────────────────────────────────
class DiscriminatorP(nn.Module):
    def __init__(self, period, kernel_size=5, stride=3):
        super().__init__()
        self.period = period
        ch = [32, 128, 512, 1024, 1024]
        self.convs = nn.ModuleList([
            weight_norm(nn.Conv2d(1,    ch[0], (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            weight_norm(nn.Conv2d(ch[0], ch[1], (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            weight_norm(nn.Conv2d(ch[1], ch[2], (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            weight_norm(nn.Conv2d(ch[2], ch[3], (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            weight_norm(nn.Conv2d(ch[3], ch[4], (kernel_size, 1), 1,           padding=(2, 0))),
        ])
        self.conv_post = weight_norm(nn.Conv2d(ch[4], 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []
        b, c, t = x.shape
        if t % self.period != 0:
            pad = self.period - (t % self.period)
            x = F.pad(x, (0, pad), "reflect")
            t = t + pad
        x = x.view(b, c, t // self.period, self.period)

        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        return x, fmap


class MultiPeriodDiscriminator(nn.Module):
    def __init__(self, periods=(2, 3, 5, 7, 11)):
        super().__init__()
        self.discriminators = nn.ModuleList([DiscriminatorP(p) for p in periods])

    def forward(self, y, y_hat):
        y_d_rs, y_d_gs, fmap_rs, fmap_gs = [], [], [], []
        for d in self.discriminators:
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r); fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g); fmap_gs.append(fmap_g)
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


# ─────────────────────────────────────────────────────────────
# Multi-Scale Discriminator
# ─────────────────────────────────────────────────────────────
class DiscriminatorS(nn.Module):
    def __init__(self):
        super().__init__()
        self.convs = nn.ModuleList([
            weight_norm(Conv1d(1,    128,  15, 1,  padding=7)),
            weight_norm(Conv1d(128,  128,  41, 2,  padding=20, groups=4)),
            weight_norm(Conv1d(128,  256,  41, 2,  padding=20, groups=16)),
            weight_norm(Conv1d(256,  512,  41, 4,  padding=20, groups=16)),
            weight_norm(Conv1d(512,  1024, 41, 4,  padding=20, groups=16)),
            weight_norm(Conv1d(1024, 1024, 41, 1,  padding=20, groups=16)),
            weight_norm(Conv1d(1024, 1024, 5,  1,  padding=2)),
        ])
        self.conv_post = weight_norm(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []
        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        return x, fmap


class MultiScaleDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList([DiscriminatorS(), DiscriminatorS(), DiscriminatorS()])
        self.meanpools = nn.ModuleList([nn.AvgPool1d(4, 2, padding=2), nn.AvgPool1d(4, 2, padding=2)])

    def forward(self, y, y_hat):
        y_d_rs, y_d_gs, fmap_rs, fmap_gs = [], [], [], []
        for i, d in enumerate(self.discriminators):
            if i != 0:
                y = self.meanpools[i - 1](y)
                y_hat = self.meanpools[i - 1](y_hat)
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r); fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g); fmap_gs.append(fmap_g)
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


# ─────────────────────────────────────────────────────────────
# Losses (standard HiFi-GAN LSGAN + feature-matching + mel L1)
# ─────────────────────────────────────────────────────────────
def feature_loss(fmap_r, fmap_g):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl - gl))
    return loss * 2


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1 - dr) ** 2)
        g_loss = torch.mean(dg ** 2)
        loss += r_loss + g_loss
    return loss


def generator_loss(disc_outputs):
    loss = 0
    for dg in disc_outputs:
        loss += torch.mean((1 - dg) ** 2)
    return loss