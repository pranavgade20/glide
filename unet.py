import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
# import torch.optim as optim
import transformer

def convert_module_to_f16(l):
    """
    Convert primitive modules to float16.
    """
    if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        l.weight.data = l.weight.data.half()
        if l.bias is not None:
            l.bias.data = l.bias.data.half()

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
            out_channels,
            attn,
            emb_channels=768,
            use_conv=False,
            use_scale_shift_norm=True,
            dropout=0.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or in_channels
        self.use_conv = use_conv
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            nn.Sequential(
                nn.GroupNorm(num_groups=32, num_channels=self.in_channels, eps=1e-5),
                nn.SiLU()
            ),
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, padding=1),
        )
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_channels,
                2 * self.out_channels if self.use_scale_shift_norm else self.out_channels
            ),
        )

        # self.group_norm = normalization(self.out_channels, swish=0.0 if use_scale_shift_norm else 1.0),
        self.group_norm = nn.GroupNorm(num_groups=32, num_channels=self.out_channels, eps=1e-5)
        self.out_layers = nn.Sequential(
            nn.SiLU(),
            nn.Dropout(self.dropout),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1),
        )

        if self.in_channels != self.out_channels:
            self.skip_connection = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=(1, 1))
        self.attn = attn
        if self.attn:
            self.attention_layer = AttentionBlock(channels=out_channels, num_head_channels=64, encoder_channels=512)

    def forward(self, x, time_emb, text_emb):
        self._emb =(time_emb, text_emb)
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C x ...] Tensor of features.
        :param time_emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        h = self.in_layers(x)

        # emb_out = self.emb_layers(time_emb)[:, :, None, None]
        emb_out = self.emb_layers(time_emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = self.group_norm(h) * (1 + scale) + shift
            h = self.out_layers(h)
        else:
            h = h + emb_out
            h = self.out_layers(self.group_norm(h))
        if self.in_channels != self.out_channels:
            x = self.skip_connection(x)
        ret = x + h

        if self.attn is True:
            ret = self.attention_layer(ret, text_emb)

        return ret


class UpResidualBlock(ResidualBlock):
    def forward(self, x, time_emb, text_emb=None):
        h = self.in_layers[0](x)  # First group norm
        h = F.interpolate(h, scale_factor=2, mode="nearest")
        h = self.group_norm(self.in_layers[1:](h))

        emb_out = self.emb_layers(time_emb)[:, :, None, None]
        if self.use_scale_shift_norm:
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = h * (1 + scale) + shift
        else:
            h = h + emb_out
        h = self.out_layers(h)

        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.in_channels != self.out_channels:
            x = self.skip_connection(x)
        return x + h


class DownResidualBlock(ResidualBlock):
    def forward(self, x, time_emb, text_emb=None):
        h = self.in_layers[0](x)  # First group norm
        h = F.avg_pool2d(h, kernel_size=2)
        h = self.group_norm(self.in_layers[1:](h))

        emb_out = self.emb_layers(time_emb)[:, :, None, None]
        if self.use_scale_shift_norm:
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = h * (1 + scale) + shift
        else:
            h = h + emb_out
        h = self.out_layers(h)

        x = F.avg_pool2d(x, kernel_size=2)
        if self.in_channels != self.out_channels:
            x = self.skip_connection(x)
        return x + h


from transformer import QKVMultiheadAttention


class QKVAttention(nn.Module):
    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv, encoder_kv=None):
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        if encoder_kv is not None:
            assert encoder_kv.shape[1] == self.n_heads * ch * 2
            ek, ev = encoder_kv.reshape(bs * self.n_heads, ch * 2, -1).split(ch, dim=1)
            k = torch.cat([ek, k], dim=-1)
            v = torch.cat([ev, v], dim=-1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)


class AttentionBlock(nn.Module):
    def __init__(
            self,
            channels,
            num_heads=1,
            num_head_channels=-1,
            encoder_channels=512,  # int
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
            self.encoder_kv = nn.Conv1d(encoder_channels, channels * 2, 1)
        self.proj_out = nn.Conv1d(channels, channels, 1)

    def forward(self, x, encoder_out=None):
        qkv = self.qkv(self.group_norm(x).reshape(x.shape[0], x.shape[1], -1))
        if encoder_out is not None:
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
            model_channels=192,
    ):
        super().__init__()
        self.model_channels = model_channels
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        def create_model_from_arch(arch_in, arch_out):
            class Wrapper(nn.Module):
                def __init__(self, module):
                    super().__init__()
                    self.module = module

                def forward(self, x, *args):
                    return self.module(x)

            in_layers = [Wrapper(nn.Conv2d(3, model_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))]
            stack = [model_channels]

            arch_cp = copy.deepcopy(arch_in)
            for block_dict in arch_cp:
                if block_dict.pop('change_size', False):
                    in_layers.append(DownResidualBlock(**block_dict))
                else:
                    in_layers.append(ResidualBlock(**block_dict))

            stack += [d['out_channels'] for d in arch_cp]

            num_middle_channels = arch_in[-1]["out_channels"]
            middle_layers = []
            middle_layers.append(ResidualBlock(num_middle_channels, num_middle_channels, False))
            middle_layers.append(
                Wrapper(AttentionBlock(channels=num_middle_channels, num_head_channels=64, encoder_channels=512)))
            middle_layers.append(ResidualBlock(num_middle_channels, num_middle_channels, False))

            out_layers = []
            for i, block_dict in enumerate(reversed(out_arch)):
                block_dict["in_channels"] += stack.pop()
                if block_dict.pop('change_size', False):
                    a = [ResidualBlock(**block_dict)]
                    block_dict["in_channels"] = block_dict["out_channels"]
                    block_dict["attn"] = False
                    a.append(UpResidualBlock(**block_dict))
                    out_layers.append(nn.Sequential(*a))
                else:
                    out_layers.append(ResidualBlock(**block_dict))

            return in_layers, middle_layers, out_layers

        #
        in_arch = [
            {'in_channels': model_channels, 'out_channels': model_channels, 'attn': False},
            {'in_channels': model_channels, 'out_channels': model_channels, 'attn': False},
            {'in_channels': model_channels, 'out_channels': model_channels, 'attn': False},
            {'in_channels': model_channels, 'out_channels': model_channels, 'attn': False, 'change_size': True},
            {'in_channels': model_channels, 'out_channels': 2 * model_channels, 'attn': True},
            {'in_channels': 2 * model_channels, 'out_channels': 2 * model_channels, 'attn': True},
            {'in_channels': 2 * model_channels, 'out_channels': 2 * model_channels, 'attn': True},
            {'in_channels': 2 * model_channels, 'out_channels': 2 * model_channels, 'attn': False, 'change_size': True},
            {'in_channels': 2 * model_channels, 'out_channels': 3 * model_channels, 'attn': True},
            {'in_channels': 3 * model_channels, 'out_channels': 3 * model_channels, 'attn': True},
            {'in_channels': 3 * model_channels, 'out_channels': 3 * model_channels, 'attn': True},
            {'in_channels': 3 * model_channels, 'out_channels': 3 * model_channels, 'attn': False, 'change_size': True},
            {'in_channels': 3 * model_channels, 'out_channels': 4 * model_channels, 'attn': True},
            {'in_channels': 4 * model_channels, 'out_channels': 4 * model_channels, 'attn': True},
            {'in_channels': 4 * model_channels, 'out_channels': 4 * model_channels, 'attn': True},
        ]

        out_arch = [
            {'in_channels': model_channels, 'out_channels': model_channels, 'attn': False},
            {'in_channels': model_channels, 'out_channels': model_channels, 'attn': False},
            {'in_channels': model_channels, 'out_channels': model_channels, 'attn': False},
            {'in_channels': 2 * model_channels, 'out_channels': model_channels, 'attn': False},
            {'in_channels': 2 * model_channels, 'out_channels': 2 * model_channels, 'attn': True, 'change_size': True},
            {'in_channels': 2 * model_channels, 'out_channels': 2 * model_channels, 'attn': True},
            {'in_channels': 2 * model_channels, 'out_channels': 2 * model_channels, 'attn': True},
            {'in_channels': 3 * model_channels, 'out_channels': 2 * model_channels, 'attn': True},
            {'in_channels': 3 * model_channels, 'out_channels': 3 * model_channels, 'attn': True, 'change_size': True},
            {'in_channels': 3 * model_channels, 'out_channels': 3 * model_channels, 'attn': True},
            {'in_channels': 3 * model_channels, 'out_channels': 3 * model_channels, 'attn': True},
            {'in_channels': 4 * model_channels, 'out_channels': 3 * model_channels, 'attn': True},
            {'in_channels': 4 * model_channels, 'out_channels': 4 * model_channels, 'attn': True, 'change_size': True},
            {'in_channels': 4 * model_channels, 'out_channels': 4 * model_channels, 'attn': True},
            {'in_channels': 4 * model_channels, 'out_channels': 4 * model_channels, 'attn': True},
            {'in_channels': 4 * model_channels, 'out_channels': 4 * model_channels, 'attn': True},
        ]

        in_layers, middle_layers, out_layers = create_model_from_arch(in_arch, out_arch)

        self.in_layers = nn.Sequential(*in_layers)
        self.middle_layers = nn.Sequential(*middle_layers)
        self.out_layers = nn.Sequential(*out_layers)

        self.out = nn.Sequential(
            nn.GroupNorm(num_groups=32, num_channels=model_channels),
            nn.Conv2d(model_channels, 6, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )


class Text2Im(UNetModel):
    def __init__(
            self,
            tokenizer,
            text_ctx=128,
            xf_width=512,
            xf_layers=16,
            xf_heads=8,
            xf_final_ln=True,
            xf_padding=True,
    ):
        super().__init__()
        self.text_ctx = text_ctx
        self.xf_width = xf_width
        self.xf_layers = xf_layers
        self.xf_heads = xf_heads
        self.xf_final_ln = xf_final_ln
        self.xf_padding = xf_padding

        self.positional_embedding = nn.Parameter(torch.empty(text_ctx, xf_width, dtype=torch.float32))
        self.padding_embedding = nn.Parameter(torch.empty(text_ctx, xf_width, dtype=torch.float32))
        self.tokenizer = tokenizer
        self.transformer = transformer.Transformer(
            text_ctx, xf_width, xf_layers, xf_heads
        )

        self.final_ln = nn.LayerNorm(xf_width)
        self.token_embedding = nn.Embedding(self.tokenizer.n_vocab, xf_width)
        self.transformer_proj = nn.Linear(xf_width, 4 * self.model_channels)

        self.cache_text_emb = True
        self.cache = None
        self.dtype = torch.float16

    def get_text_emb(self, tokens, mask):
        """from https://github.com/openai/glide-text2im/blob/9cc8e563851bd38f5ddb3e305127192cb0f02f5c/glide_text2im/text2im_model.py#L89"""
        assert tokens is not None

        if self.cache_text_emb and self.cache is not None:
            assert (
                    tokens == self.cache["tokens"]
            ).all(), f"Tokens {tokens.cpu().numpy().tolist()} do not match cache {self.cache['tokens'].cpu().numpy().tolist()}"
            return self.cache

        xf_in = self.token_embedding(tokens.long())
        xf_in = xf_in + self.positional_embedding[None]
        if self.xf_padding:
            assert mask is not None
            xf_in = torch.where(mask[..., None], xf_in, self.padding_embedding[None])
        xf_out = self.transformer(xf_in.to(self.dtype))
        if self.final_ln is not None:
            xf_out = self.final_ln(xf_out)
        xf_proj = self.transformer_proj(xf_out[:, -1])
        xf_out = xf_out.permute(0, 2, 1)  # NLC -> NCL

        outputs = dict(xf_proj=xf_proj, xf_out=xf_out)

        if self.cache_text_emb:
            self.cache = dict(
                tokens=tokens,
                xf_proj=xf_proj.detach(),
                xf_out=xf_out.detach() if xf_out is not None else None,
            )

        return outputs

    def forward(self, x, timesteps, tokens, mask):
        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        if self.xf_width:
            text_outputs = self.get_text_emb(tokens, mask)
            xf_proj, xf_out = text_outputs["xf_proj"], text_outputs["xf_out"]
            emb = emb + xf_proj.to(emb)
        else:
            xf_out = None
        h = x.type(self.dtype)
        for module in self.in_layers:
            h = module(h, emb, xf_out)
            hs.append(h)

        for module in self.middle_layers:
            h = module(h, emb, xf_out)

        for module in self.out_layers:
            h = torch.cat([h, hs.pop()], dim=1)
            if isinstance(module, nn.Sequential):
                for l in module:
                    h = l(h, emb, xf_out)
            else:
                h = module(h, emb, xf_out)
        h = h.type(x.dtype)
        h = self.out(h)
        return h

    def del_cache(self):
        self.cache = None

    def convert_to_fp16(self):
        self.in_layers.apply(convert_module_to_f16)
        self.middle_layers.apply(convert_module_to_f16)
        self.out_layers.apply(convert_module_to_f16)
        if self.xf_width:
            self.transformer.apply(convert_module_to_f16)
            self.transformer_proj.to(torch.float16)
            self.token_embedding.to(torch.float16)
            self.positional_embedding.to(torch.float16)
            if self.xf_padding:
                self.padding_embedding.to(torch.float16)

def timestep_embedding(timesteps, dim, max_period=10000):
    """
    copied from https://github.com/openai/glide-text2im/blob/9cc8e563851bd38f5ddb3e305127192cb0f02f5c/glide_text2im/nn.py#L87
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding
