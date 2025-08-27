import functools
import torch.nn as nn

import register
from classification import utils
from nn_module.conv.convs import Conv2d
from nn_module.norm.dispatch_norm import DispatchNorm


@register.NAME_TO_CONVS.register("ConvNormAct")
class ConvNormAct(nn.Module):
    """
    Building block that is generally used in most convolutional neural networks.
    The ConvNormAct block is composed of one sub-block conv, one normalization layer and one activation layer.
    Example1:
        By setting conv to Conv2d, norm to BatchNorm2d and act to ReLU, the ConvNormAct block is simply a standard
        conv-bn-relu block.
    Example2:
        By setting conv to ResBlock(nn_module/conv/res_block/ResBlock), norm to BatchNorm2d and act to ReLU,
        the ConvNormAct block is a residual block defined in ResNet.
    """

    def __init__(self, in_channels, out_channels, stride, norm=nn.BatchNorm2d, norm_kwargs=None, act=nn.ReLU, dropout=0.0, pre_act=False, conv=Conv2d):
        super().__init__()

        # Create a mutable copy of the norm_kwargs dictionary or an empty one
        final_norm_kwargs = (norm_kwargs or {}).copy()

        # Determine the number of channels for the normalization layer
        channels = in_channels if pre_act else out_channels

        # Set default values for channel arguments. This makes it compatible
        # with both BatchNorm2d (needs num_features) and GroupNorm (needs num_channels).
        # The user can override this in their YAML if needed.
        if norm == nn.GroupNorm:
            # GroupNorm requires 'num_channels'. It's an error if the user
            # hasn't provided 'num_groups' in their YAML.
            if 'num_channels' not in final_norm_kwargs:
                final_norm_kwargs['num_channels'] = channels
        elif norm == nn.BatchNorm2d or norm == nn.InstanceNorm2d:
            # These layers require 'num_features'.
            if 'num_features' not in final_norm_kwargs:
                final_norm_kwargs['num_features'] = channels

        if pre_act:
            self.convs = nn.Sequential(
                DispatchNorm(norm, **final_norm_kwargs),
                act(),
                conv(in_channels=in_channels, out_channels=out_channels, stride=stride),
                nn.Dropout(dropout)
            )
        else:
            self.convs = nn.Sequential(
                conv(in_channels=in_channels, out_channels=out_channels, stride=stride),
                DispatchNorm(norm, **final_norm_kwargs),
                act(),
                nn.Dropout(dropout)
            )

    def forward(self, x):
        return self.convs(x)

    @staticmethod
    def get_conv(configs):
        conv = register.get_conv(configs)

        norm_config = configs["norm"]
        if isinstance(norm_config, dict):
            # This handles the new format, e.g., {GroupNorm: {num_groups: 32}}
            norm_name = list(norm_config.keys())[0]
            norm_kwargs = norm_config[norm_name]
            # Your register should be able to get the class from the name
            norm_class = register.get_norm(norm_name) 
        else:
            # This handles the old format, e.g., "BatchNorm2d"
            norm_class = register.get_norm(norm_config)
            norm_kwargs = None

        act = register.get_activation(configs["act"])

        default_params = {
            "norm"       : norm_class,
            "norm_kwargs": norm_kwargs, # <-- ADD THIS
            "act"    : act,
            "dropout": 0.0,
            "pre_act": False,
            "conv"   : conv,
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

        default_params = utils.set_params(default_params, configs, excluded_keys=["norm", "act", "conv"])
        conv = functools.partial(ConvNormAct, **default_params)
        return conv