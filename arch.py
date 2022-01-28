import copy

from unet import *

in_arch = [
    {'in_channels': 192, 'out_channels': 192, 'attn': False},
    {'in_channels': 192, 'out_channels': 192, 'attn': False},
    {'in_channels': 192, 'out_channels': 192, 'attn': False},
    {'in_channels': 192, 'out_channels': 192, 'attn': False, 'change_size': True},
    {'in_channels': 192, 'out_channels': 2 * 192, 'attn': True},
    {'in_channels': 2 * 192, 'out_channels': 2 * 192, 'attn': True},
    {'in_channels': 2 * 192, 'out_channels': 2 * 192, 'attn': True},
    {'in_channels': 2 * 192, 'out_channels': 2 * 192, 'attn': False, 'change_size': True},
    {'in_channels': 2 * 192, 'out_channels': 3 * 192, 'attn': True},
    {'in_channels': 3 * 192, 'out_channels': 3 * 192, 'attn': True},
    {'in_channels': 3 * 192, 'out_channels': 3 * 192, 'attn': True},
    {'in_channels': 3 * 192, 'out_channels': 3 * 192, 'attn': False, 'change_size': True},
    {'in_channels': 3 * 192, 'out_channels': 4 * 192, 'attn': True},
    {'in_channels': 4 * 192, 'out_channels': 4 * 192, 'attn': True},
    {'in_channels': 4 * 192, 'out_channels': 4 * 192, 'attn': True},
]


def create_model_from_arch(architecture):
    ret = [nn.Conv2d(3, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))]
    stack = [192]
    # create the input modules

    arch_cp = copy.deepcopy(architecture)
    for block_dict in arch_cp:
        if block_dict.pop('change_size', False):
            ret.append(DownResidualBlock(**block_dict))
        else:
            ret.append(ResidualBlock(**block_dict))

    num_middle_channels = architecture[-1]["out_channels"]

    ret.append(ResidualBlock(num_middle_channels, num_middle_channels, False))
    ret.append(AttentionBlock(channels=num_middle_channels, num_head_channels=64, encoder_channels=512))
    ret.append(ResidualBlock(num_middle_channels, num_middle_channels, False))

    out_arch = [{
        'in_channels': d['out_channels'],
        'out_channels': d['in_channels'],
        'attn': d['attn'],
        'change_size': 'change_size' in d and d['change_size']
    } for d in reversed(architecture)]

    out_arch.insert(0, out_arch[0])

    for block_dict in out_arch:
        if block_dict.pop('change_size', False):
            ret.append(UpResidualBlock(**block_dict))
        else:
            ret.append(ResidualBlock(**block_dict))

    return ret

if __name__ == '__main__':
    arch = create_model_from_arch(in_arch)
    model = nn.Sequential(*arch)
    print(model)