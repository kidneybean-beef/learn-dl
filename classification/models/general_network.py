import torch.nn as nn
from collections import OrderedDict

import register
from classification import utils
from classification.models.plainnet import PlainNet
from nn_module.norm.dispatch_norm import DispatchNorm


@register.name_to_model.register("GNetwork")
class GeneralNetwork(nn.Module):

    def __init__(self, in_channels, hidden_channels, n_blocks_list, stride_list, stride_factor_list,
                 num_classes, convs, fcs=None, last_norm=nn.BatchNorm2d, last_norm_kwargs=None, last_act=nn.ReLU, out_feats=None):
        super().__init__()
        assert len(n_blocks_list) > 0
        assert len(n_blocks_list) == len(stride_list) == len(stride_factor_list) == len(convs)
        self.n = len(n_blocks_list)
        self.out_feats = out_feats

        self.convs, out_channels = self.make_backbone(n_blocks_list, stride_list, in_channels, hidden_channels,
                                                      stride_factor_list, convs)
        
        # Create a mutable copy of the norm_kwargs dictionary or an empty one
        last_norm_final_kwargs = (last_norm_kwargs or {}).copy()

        # Set default values for channel arguments. This makes it compatible
        # with both BatchNorm2d (needs num_features) and GroupNorm (needs num_channels).
        # The user can override this in their YAML if needed.
        if last_norm == nn.GroupNorm:
            # GroupNorm requires 'num_channels'. It's an error if the user
            # hasn't provided 'num_groups' in their YAML.
            if 'num_channels' not in last_norm_final_kwargs:
                last_norm_final_kwargs['num_channels'] = out_channels
        elif last_norm == nn.BatchNorm2d or last_norm == nn.InstanceNorm2d:
            # These layers require 'num_features'.
            if 'num_features' not in last_norm_final_kwargs:
                last_norm_final_kwargs['num_features'] = out_channels
        self.last_norm = DispatchNorm(last_norm, **last_norm_final_kwargs)
        self.last_act = last_act()

        # For binary classification task, we use BCE loss so only one output logit is needed.
        self.num_classes = num_classes
        if num_classes == 2:
            num_classes = 1
        if num_classes > 0:
            self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
            if fcs is None:
                self.fcs = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(out_channels, num_classes)
                )
            else:
                self.fcs = [conv() for conv in fcs]
                self.fcs = nn.Sequential(*self.fcs)

    @staticmethod
    def make_backbone(n_blocks_list, stride_list, in_channels, hidden_channels, stride_factor_list, conv_list):
        convs = []

        for i, (n_blocks, stride, stride_factor, conv) in enumerate(zip(n_blocks_list, stride_list, stride_factor_list, conv_list)):
            if i != 0:
                hidden_channels = int(in_channels * stride_factor)
            convs += PlainNet.make_plain_part(in_channels, hidden_channels, stride, n_blocks, conv)
            in_channels = hidden_channels

        convs = nn.Sequential(*convs)
        return convs, hidden_channels

    def forward(self, x):
        if self.out_feats is None or self.out_feats == "None":
            output = self.last_norm(self.convs(x))
            output = self.last_act(output)
            if self.num_classes > 0:
                output = self.avg_pool(output)
                output = self.fcs(output)
                output = output.view(output.size(0), -1)
            return output
        else:
            outputs = OrderedDict()
            for i, (name, layer) in enumerate(self.convs.named_children()):
                x = layer(x)
                if i in self.out_feats:
                    outputs[f"layer_{i}"] = x
            output = self.last_norm(x)
            output = self.last_act(output)
            if self.num_classes > 0:
                output = self.avg_pool(output)
                output = self.fcs(output)
                output = output.view(output.size(0), -1)
            outputs["layer_last"] = output
            return outputs

    @staticmethod
    def make_network(configs):
        config_convs = configs["convs"]
        conv_list = []
        for i in range(len(config_convs)):
            if f"conv{i}" not in config_convs:
                raise ValueError(f"The key \"conv{i}\" is not specified.")
            else:
                conv_list.append(register.get_conv(config_convs, f"conv{i}"))

        fc_list = None
        if "fcs" in configs:
            config_fcs = configs["fcs"]
            fc_list = []
            for i in range(len(config_fcs)):
                if f"fc{i}" not in config_fcs:
                    raise ValueError(f"The key \"fc{i}\" is not specified.")
                else:
                    fc_list.append(register.get_conv(config_fcs, f"fc{i}"))


        norm_config = configs["last_norm"]
        if isinstance(norm_config, dict):
            # This handles the new format, e.g., {GroupNorm: {num_groups: 32}}
            norm_name = list(norm_config.keys())[0]
            norm_kwargs = norm_config[norm_name]
            # Your register should be able to get the class from the name
            norm = register.get_norm(norm_name) 
        else:
            # This handles the old format, e.g., "BatchNorm2d"
            norm = register.get_norm(norm_config)
            norm_kwargs = None
        act = register.get_activation(configs["last_act"])

        default_params = {
            "in_channels"       : 3,
            "hidden_channels"   : 64,
            "n_blocks_list"     : [2, 2, 2, 2],
            "stride_list"       : [2, 1, 1, 1],
            "stride_factor_list": [2, 2, 2, 2],
            "num_classes"       : 10,
            "convs"             : conv_list,
            "fcs"               : fc_list,
            "last_norm"         : norm,
            "last_norm_kwargs"  : norm_kwargs,
            "last_act"          : act,
            "out_feats"         : None,
        }
        default_params = utils.set_params(default_params, configs, excluded_keys=["convs", "fcs", "last_norm", "last_norm_kwargs", "last_act"])
        return GeneralNetwork(**default_params)