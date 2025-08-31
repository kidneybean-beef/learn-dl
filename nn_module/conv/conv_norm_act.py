import functools
import torch
import torch.nn as nn

import register
from classification import utils
from nn_module.conv.convs import Conv2d, ConcatConv2d
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

        channels = in_channels if pre_act else out_channels

        # Set default values for channel arguments. This makes it compatible
        # with both BatchNorm2d (needs num_features) and GroupNorm (needs num_channels).
        # The user can override this in their YAML if needed.
        final_norm_kwargs = (norm_kwargs or {}).copy()
        if norm == nn.GroupNorm:
            # GroupNorm requires 'num_channels'. It's an error if the user
            # hasn't provided 'num_groups' in their YAML.
            if 'num_channels' not in final_norm_kwargs:
                final_norm_kwargs['num_channels'] = channels
        elif norm == nn.BatchNorm2d or norm == nn.InstanceNorm2d:
            # These layers require 'num_features'.
            if 'num_features' not in final_norm_kwargs:
                final_norm_kwargs['num_features'] = channels

        self.conv_layer = conv(in_channels=in_channels, out_channels=out_channels, stride=stride)
        self.norm_layer = DispatchNorm(norm, **final_norm_kwargs)
        self.act_layer = act()
        self.dropout_layer = nn.Dropout(dropout)

        self.pre_act = pre_act
        if self.pre_act:
            self.convs = self._forward_pre_act
        else:
            self.convs = self._forward_post_act

    def _forward_pre_act(self, x):
        # Path: Norm -> Act -> Conv -> Dropout
        out = self.norm_layer(x)
        out = self.act_layer(out)
        out = self.conv_layer(out)
        out = self.dropout_layer(out)
        return out

    def _forward_post_act(self, x):
        # Path: Conv -> Norm -> Act -> Dropout
        out = self.conv_layer(x)
        out = self.norm_layer(out)
        out = self.act_layer(out)
        out = self.dropout_layer(out)
        return out

    def forward(self, x):
        return self.convs(x)

    @staticmethod
    def get_conv(configs):
        if "conv" not in configs:
            raise ValueError("TimeAwareConvNormAct config must have a 'conv' key.")
    
        conv_builder = register.get_conv(configs, "conv")

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
            "norm_kwargs": norm_kwargs,
            "act"    : act,
            "dropout": 0.0,
            "pre_act": False,
            "conv"   : conv_builder,
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
    
@register.NAME_TO_CONVS.register("TimeAwareConvNormAct")
class TimeAwareConvNormAct(ConvNormAct):

    def __init__(self, in_channels, out_channels, stride, norm=nn.BatchNorm2d, norm_kwargs=None, act=nn.ReLU, dropout=0.0, pre_act=False, conv=ConcatConv2d):
        
        # Call the __init__ of the parent class (Conv2d) with the modified
        # in_channels and all other original arguments. This reuses all the
        # logic for spectral norm, bias, padding_mode, etc.
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            norm=norm,
            norm_kwargs=norm_kwargs,
            act=act,
            dropout=dropout,
            pre_act=pre_act,
            conv=conv
        )

        if self.pre_act:
            self.convs = self._forward_pre_act
        else:
            self.convs = self._forward_post_act

    def _forward_pre_act(self, t, x):
        # Path: Norm -> Act -> Conv -> Dropout
        out = self.norm_layer(x)
        out = self.act_layer(out)
        out = self.conv_layer(t, out)
        out = self.dropout_layer(out)
        return out

    def _forward_post_act(self, t, x):
        # Path: Conv -> Norm -> Act -> Dropout
        out = self.conv_layer(t, x)
        out = self.norm_layer(out)
        out = self.act_layer(out)
        out = self.dropout_layer(out)
        return out

    def forward(self, t, x):
        return self.convs(t, x)

    @staticmethod
    def get_conv(configs):
        if "conv" not in configs:
            raise ValueError("TimeAwareConvNormAct config must have a 'conv' key.")
        conv_builder = register.get_conv(configs, "conv")

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
            "norm_kwargs": norm_kwargs,
            "act"    : act,
            "dropout": 0.0,
            "pre_act": False,
            "conv"   : conv_builder,
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
        conv = functools.partial(TimeAwareConvNormAct, **default_params)
        return conv