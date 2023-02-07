import torch
import torch.nn as nn


def vectorize(x):
    return x.view(x.shape[0], -1)


class ConvLayer(torch.nn.Sequential):
    def __init__(self, conv, in_channels, out_channels, kernel_size, stride, padding=None, dilation=1, norm=None, act=None, bias=True):
        super(ConvLayer, self).__init__()
        # padding = padding or kernel_size // 2
        padding = padding or dilation * (kernel_size - 1) // 2
        self.add_module('conv2d', conv(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation, bias=bias))
        if norm is not None:
            self.add_module('norm', norm(out_channels))
            # self.add_module('norm', norm(out_channels, track_running_stats=True))
        if act is not None:
            self.add_module('act', act)
