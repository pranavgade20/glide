from turtle import forward
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

        self.in_layers = nn.Sequential([
            nn.GroupNorm(num_groups=32, num_channels=self.channels, eps=1e-5),
            nn.SiLU(),
            nn.Conv2d(self.channels, self.out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=self.out_channels, eps=1e-5),
        ])

        self.emb_layers = nn.Sequential([
            nn.SiLU(),
            nn.Linear(
                emb_channels,
                2*self.out_channels if self.use_scale_shift_norm else self.out_channels
                ),
        ])

        self.out_layers = nn.Sequential([
            nn.SiLU(),
            nn.Dropout(self.dropout),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1),
        ])

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
    def __init__(
        self,
        channels,
        emb_channels,
        out_channels,
        use_conv=False,
        use_scale_shift_norm=False,
        dropout=0
        ):
        super().__init__(channels, emb_channels, out_channels, use_conv, use_scale_shift_norm, dropout):


