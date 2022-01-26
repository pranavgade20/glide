import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class ResidualBlock(nn.Module):
    """
    A residual block that can optionally change the number of channels.

    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    """
    def __init__(
        self,
        channels,
        emb_channels,
        out_channels,
        use_conv=False,
        use_scale_shift_norm=False,
        dropout=0.0,
    ):
        super().__init__()
        assert out_channels is None
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            nn.GroupNorm(num_groups=32, num_channels=self.channels, eps=1e-5),
            nn.SiLU(),
            nn.Conv2d(self.channels, self.out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=self.out_channels, eps=1e-5),
        )

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_channels,
                2*self.out_channels if self.use_scale_shift_norm else self.out_channels
                ),
        )

        self.out_layers = nn.Sequential(
            nn.SiLU(),
            nn.Dropout(self.dropout),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb)
        assert len(emb_out.shape) == len(h.shape)
        if self.use_scale_shift_norm:
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = h * (1 + scale) + shift
        else:
            h = h + emb_out
        h = self.out_layers(h)
        return x + h

class UpResidualBlock(ResidualBlock):
    def forward(self, x, emb):
        h = self.in_layers[:1](x)
        h = F.interpolate(h, scale_factor=2, mode="nearest")
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        h = self.in_layers[1:](x)
        emb_out = self.emb_layers(emb)
        assert len(emb_out.shape) == len(h.shape)
        if self.use_scale_shift_norm:
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = h * (1 + scale) + shift
        else:
            h = h + emb_out
        h = self.out_layers(h)
        return x + h


class DownResidualBlock(ResidualBlock):
    def forward(self, x, emb):
        h = self.in_layers[:1](x)
        h = F.avg_pool2d(h, kernel_size=(1, 2, 2))
        x = F.avg_pool2d(x, kernel_size=(1, 2, 2))
        h = self.in_layers[1:](x)
        emb_out = self.emb_layers(emb)
        assert len(emb_out.shape) == len(h.shape)
        if self.use_scale_shift_norm:
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = h * (1 + scale) + shift
        else:
            h = h + emb_out
        h = self.out_layers(h)
        return x + h


from transformer import QKVMultiheadAttention


class QKVAttention(QKVMultiheadAttention):
    def forward(self, qkv, encoder_kv=None):
        bs, n_ctx, width = qkv.shape
        attn_ch = width // self.n_heads // 3
        scale = 1 / math.sqrt(math.sqrt(attn_ch))
        qkv = qkv.view(bs, n_ctx, self.n_heads, -1)
        q, k, v = torch.split(qkv, attn_ch, dim=-1)
        if encoder_kv is not None:
            assert encoder_kv.shape[1] == self.n_heads * attn_ch * 2
            ek, ev = encoder_kv.reshape(bs * self.n_heads, attn_ch * 2, -1).split(attn_ch, dim=1)
            k = torch.cat([ek, k], dim=-1)
            v = torch.cat([ev, v], dim=-1)

        weight = torch.einsum(
            "bthc,bshc->bhts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        wdtype = weight.dtype
        weight = torch.softmax(weight.float(), dim=-1).type(wdtype)
        return torch.einsum("bhts,bshc->bthc", weight, v).reshape(bs, n_ctx, -1)


class AttentionBlock(nn.Module):
    def __init__(
            self,
            channels,
            num_heads=1,
            num_head_channels=-1,
            use_checkpoint=False,
            encoder_channels=None,
    ):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        if num_head_channels > 0:
            assert channels % num_head_channels == 0
            self.num_heads = channels // num_head_channels

        self.group_norm = nn.GroupNorm(num_groups=32, num_channels=channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.attention = QKVAttention(self.num_heads)
        if encoder_channels:
            self.encoder_kv = nn.Conv1d(channels, channels * 2, 1)
        self.proj_out = nn.Conv1d(channels, channels, 1)

    def forward(self, x, encoder_out=None):
        qkv = self.qkv(self.group_norm(x).reshape(x.shape[0], x.shape[1], -1))
        if encoder_out:
            encoder_out = self.encoder_kv(encoder_out)
        y = self.attention(qkv, encoder_out)
        y = self.proj_out(y)
        return x + y.reshape(x.shape)
