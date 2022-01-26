import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class ResidualBlock(nn.Module):
    """
    A residual block that can optionally change the number of channels.

    :param in_channels: the number of input channels.
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
        in_channels,
        emb_channels,
        out_channels,
        use_conv=False,
        use_scale_shift_norm=False,
        dropout=0.0,
    ):
        super().__init__()
        assert out_channels is None
        self.in_channels = in_channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or in_channels
        self.use_conv = use_conv
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            nn.GroupNorm(num_groups=32, num_channels=self.in_channels, eps=1e-5),
            nn.SiLU(),
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, padding=1),
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

        emb_out = self.emb_layers(emb)[:,:,None,None]
        if self.use_scale_shift_norm:
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = h * (1 + scale) + shift
        else:
            h = h + emb_out
        h = self.out_layers(h)

        return x + h

class UpResidualBlock(ResidualBlock):
    def forward(self, x, emb):
        h = self.in_layers[0](x) # First group norm
        h = F.interpolate(h, scale_factor=2, mode="nearest")
        h = self.in_layers[1:](h)

        emb_out = self.emb_layers(emb)[:,:,None,None]
        if self.use_scale_shift_norm:
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = h * (1 + scale) + shift
        else:
            h = h + emb_out
        h = self.out_layers(h)

        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return x + h


class DownResidualBlock(ResidualBlock):
    def forward(self, x, emb):
        h = self.in_layers[0](x) # First group norm
        h = F.avg_pool2d(h, kernel_size=(1, 2, 2))
        h = self.in_layers[1:](h)

        emb_out = self.emb_layers(emb)[:,:,None,None]
        if self.use_scale_shift_norm:
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = h * (1 + scale) + shift
        else:
            h = h + emb_out
        h = self.out_layers(h)

        x = F.avg_pool2d(x, kernel_size=(1, 2, 2))
        return x + h


from transformer import QKVMultiheadAttention


class QKVAttention(QKVMultiheadAttention):
    def forward(self, qkv, encoder_kv=None):
        bs, n_ctx, width = qkv.shape
        attn_ch = width // self.n_heads // 3
        scale = 1 / math.sqrt(math.sqrt(attn_ch))
        qkv = qkv.view(bs, n_ctx, self.n_heads, -1)
        q, k, v = torch.split(qkv, attn_ch, dim=-1)
        if encoder_kv:
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

class UNetModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding.

    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    """

    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        num_classes=None,
        use_checkpoint=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        encoder_channels=None,
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = torch.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        next_channels = current_channels = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList(
            [nn.Conv2d(current_channels, next_channels, kernel_size=3, padding=1)]
        )

        input_block_channels = [current_channels]
        ds = 1 #???
        for level, mult in enumerate(channel_mult):
            current_channels, next_channels = next_channels, int(mult * model_channels)
            for _ in range(num_res_blocks):
                layers = [
                    ResidualBlock(
                        current_channels,
                        time_embed_dim,
                        dropout=dropout,
                        out_channels = next_channels,
                        use_scale_shift_norm = use_scale_shift_norm
                    )
                ]








