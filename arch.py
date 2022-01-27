import torch.nn as nn

from unet import *

d_model = 192
in_channels = 3
out_channels = 6
time_emb_channels = 128
in_arch = [
    {'in_channels': d_model, 'out_channels': d_model, 'attn': False},
    {'in_channels': d_model, 'out_channels': d_model, 'attn': False},
    {'in_channels': d_model, 'out_channels': d_model, 'attn': False},
    {'in_channels': d_model, 'out_channels': d_model, 'attn': False, 'change_size': True},
    {'in_channels': d_model, 'out_channels': 2 * d_model, 'attn': True},
    {'in_channels': 2 * d_model, 'out_channels': 2 * d_model, 'attn': True},
    {'in_channels': 2 * d_model, 'out_channels': 2 * d_model, 'attn': True},
    {'in_channels': 2 * d_model, 'out_channels': 2 * d_model, 'attn': False, 'change_size': True},
    {'in_channels': 2 * d_model, 'out_channels': 3 * d_model, 'attn': True},
    {'in_channels': 3 * d_model, 'out_channels': 3 * d_model, 'attn': True},
    {'in_channels': 3 * d_model, 'out_channels': 3 * d_model, 'attn': True},
    {'in_channels': 3 * d_model, 'out_channels': 3 * d_model, 'attn': True},
    {'in_channels': 3 * d_model, 'out_channels': 3 * d_model, 'attn': False, 'change_size': True},
    {'in_channels': 3 * d_model, 'out_channels': 4 * d_model, 'attn': True},
    {'in_channels': 4 * d_model, 'out_channels': 4 * d_model, 'attn': True},
    {'in_channels': 4 * d_model, 'out_channels': 4 * d_model, 'attn': True},
]

def create_model_from_arch(architecture, text_emb_channels = 512):
    ret = []
    # create the input modules

    for block_dict in architecture:
        attn = block_dict.get('attn', True)
        change_size = block_dict.get("change_size", False)
        in_channels = block_dict['in_channels']
        out_channels = block_dict['out_channels']
        assert not (attn and change_size)
        if change_size is True:
            ret.append(DownResidualBlock(in_channels, text_emb_channels, out_channels))
        else:
            ret.append(ResidualBlock(in_channels, text_emb_channels, out_channels, attn=attn))

    num_middle_channels = architecture[-1]["out_channels"]

    # ret.append(ResidualBlock(num_middle_channels, ))

    return ret