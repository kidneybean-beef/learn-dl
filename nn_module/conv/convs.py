import functools
import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm as SN

import register
from classification import utils


@register.NAME_TO_CONVS.register("Conv2d")
class Conv2d(nn.Module):
    """
    The meta-build block of neural network.
    This is a wrapper of `torch.nn.Conv2d`.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0, dilation=1, groups=1, bias=True,
                 padding_mode="zeros", spectral_norm=False):
        """
        Parameters
        ----------
        in_channels : int
            The number of input channels.
        out_channels : int
            The number of output channels.
        kernel_size : int or tuple of int
            The size of the kernel.
        stride : int or tuple of int
            The stride of the kernel.
        padding : int or tuple of int
            The padding of the input.
        dilation : int or tuple of int
            The dilation of the kernel.
        groups : int
            The number of groups.
        bias : bool
            Whether to use bias.
        padding_mode : str
            The padding mode.
        spectral_norm : bool
            Whether to use spectral normalization.
        """
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation,
                              groups=groups, bias=bias, padding_mode=padding_mode)
        if spectral_norm:
            self.conv = SN(self.conv)

    def forward(self, x):
        return self.conv(x)

    @staticmethod
    def get_conv(configs):
        default_params = {
            "kernel_size"  : 3,
            "padding"      : 0,
            "dilation"     : 1,
            "groups"       : 1,
            "bias"         : True,
            "padding_mode" : "zeros",
            "spectral_norm": False,
        }

        # In most cases, we do not need to set the following parameters
        # as the network will infer them using stride and stride factor
        # of each layer.
        # The only reason we need to set them is when we want to use
        # 1x1 conv layers as fully connected layers.
        if "in_channels" in configs:
            default_params["in_channels"] = configs["in_channels"]
        if "out_channels" in configs:
            default_params["out_channels"] = configs["out_channels"]
        if "stride" in configs:
            default_params["stride"] = configs["stride"]

        default_params = utils.set_params(default_params, configs)
        conv = functools.partial(Conv2d, **default_params)
        return conv
    
@register.NAME_TO_CONVS.register("ConcatConv2d")
class ConcatConv2d(Conv2d):
    """
    A time-aware convolutional layer that concatenates a time channel 't'
    to the input tensor 'x' before applying the convolution.

    This class inherits from the standard Conv2d wrapper to reuse its
    initialization logic (spectral norm, etc.) and factory method structure.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, **kwargs):
        """
        Initializes the layer. It adds 1 to the `in_channels` before passing
        all arguments to the parent Conv2d's constructor.
        """
        # The actual convolution will have one extra input channel for time.
        parent_in_channels = in_channels + 1
        
        # Call the __init__ of the parent class (Conv2d) with the modified
        # in_channels and all other original arguments. This reuses all the
        # logic for spectral norm, bias, padding_mode, etc.
        super().__init__(
            in_channels=parent_in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            **kwargs
        )

    def forward(self, t, x):
        """
        The forward signature required by the ODEFunc.
        It prepares the input and then calls the parent's forward method.
        """
        # Create the time tensor, making sure it matches the device and dtype of x.
        tt = t * torch.ones(x.shape[0], 1, x.shape[2], x.shape[3], 
                            dtype=x.dtype, device=x.device)
        
        # Concatenate along the channel dimension (dim=1).
        ttx = torch.cat([tt, x], 1)
        
        # Call the parent's forward method (which just runs self.conv)
        # with the new time-concatenated tensor.
        return super().forward(ttx)

    @staticmethod
    def get_conv(configs):
        """
        Factory method for YAML parsing. This is nearly identical to the
        parent's factory, but it crucially points to its own class.
        """
        # This reuses the same logic as the parent's factory.
        default_params = {
            "kernel_size"  : 3,
            "padding"      : 0,
            "dilation"     : 1,
            "groups"       : 1,
            "bias"         : True,
            "padding_mode" : "zeros",
            "spectral_norm": False,
        }

        if "in_channels" in configs:
            default_params["in_channels"] = configs["in_channels"]
        if "out_channels" in configs:
            default_params["out_channels"] = configs["out_channels"]
        if "stride" in configs:
            default_params["stride"] = configs["stride"]

        default_params = utils.set_params(default_params, configs)
        
        # The only change from the parent's factory is the class name here.
        conv = functools.partial(ConcatConv2d, **default_params)
        return conv


@register.NAME_TO_CONVS.register("Conv2dTranspose")
class Conv2dTranspose(nn.Module):
    """
    The meta-build block of neural network.
    This is a wrapper of `torch.nn.ConvTranspose2d`.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0, dilation=1, groups=1, bias=True,
                 padding_mode="zeros", spectral_norm=False):
        """
        Parameters
        ----------
        in_channels : int
            The number of input channels.
        out_channels : int
            The number of output channels.
        kernel_size : int or tuple of int
            The size of the kernel.
        stride : int or tuple of int
            The stride of the kernel.
        padding : int or tuple of int
            The padding of the input.
        dilation : int or tuple of int
            The dilation of the kernel.
        groups : int
            The number of groups.
        bias : bool
            Whether to use bias.
        padding_mode : str
            The padding mode.
        spectral_norm : bool
            Whether to use spectral normalization.
        """
        super().__init__()
        self.conv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=bias, padding_mode=padding_mode)
        if spectral_norm:
            self.conv = SN(self.conv)

    def forward(self, x):
        return self.conv(x)

    @staticmethod
    def get_conv(configs):
        default_params = {
            "kernel_size"  : 3,
            "padding"      : 0,
            "dilation"     : 1,
            "groups"       : 1,
            "bias"         : True,
            "padding_mode" : "zeros",
            "spectral_norm": False,
        }
        default_params = utils.set_params(default_params, configs)
        conv = functools.partial(Conv2dTranspose, **default_params)
        return conv


@register.NAME_TO_CONVS.register("Conv2dUp")
class Conv2dUp(nn.Module):
    """
    The meta-build block of neural network.
    This is an upsample block consisting of `torch.nn.Upsample` and `torch.nn.Conv2d`.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0, dilation=1, groups=1, bias=True,
                 padding_mode="zeros", mode="nearest", spectral_norm=False):
        """
        Parameters
        ----------
        in_channels : int
            The number of input channels.
        out_channels : int
            The number of output channels.
        kernel_size : int or tuple of int
            The size of the kernel.
        stride : int or tuple of int
            The stride of the kernel.
        padding : int or tuple of int
            The padding of the input.
        dilation : int or tuple of int
            The dilation of the kernel.
        groups : int
            The number of groups.
        bias : bool
            Whether to use bias.
        padding_mode : str
            The padding mode.
        mode : str
            The upsampling algorithm: one of ``'nearest'``,
            ``'linear'``, ``'bilinear'``, ``'bicubic'`` and ``'trilinear'``.
            Default: ``'nearest'``
        spectral_norm : bool
            Whether to use spectral normalization.
        """
        super().__init__()
        self.up = nn.Upsample(scale_factor=stride, mode=mode)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=padding, dilation=dilation,
                              groups=groups, bias=bias, padding_mode=padding_mode)
        if spectral_norm:
            self.conv = SN(self.conv)

    def forward(self, x):
        x = self.up(x)
        return self.conv(x)

    @staticmethod
    def get_conv(configs):
        default_params = {
            "kernel_size"  : 3,
            "padding"      : 0,
            "dilation"     : 1,
            "groups"       : 1,
            "bias"         : True,
            "padding_mode" : "zeros",
            "mode"         : "nearest",
            "spectral_norm": False,
        }
        default_params = utils.set_params(default_params, configs)
        conv = functools.partial(Conv2dUp, **default_params)
        return conv