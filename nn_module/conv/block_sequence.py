import functools
import torch.nn as nn

import register
from classification import utils
from nn_module.conv.conv_norm_act import ConvNormAct
from nn_module.conv.res_block import ResBlock


@register.NAME_TO_CONVS.register("BlockSequence")
class BlockSequence(nn.Module):
    """
    A framework class that generates a sequence of blocks that is guaranteed
    to preserve the spatial size and channel count of the input tensor.
    """

    def __init__(self, channels, stride, num_blocks, block_config):
        """
        Args:
            channels (int): The constant number of channels to maintain throughout the sequence.
            num_blocks (int): The number of times to repeat the block.
            block_config (dict): The single template configuration for the block to be repeated.
        """
        super().__init__()
        
        block_name = list(block_config.keys())[0]
        specific_configs = block_config[block_name]
        
        # Get the factory method (e.g., ResBlock.get_conv) from your register
        block_factory = register.NAME_TO_CONVS[block_name].get_conv
        
        # Create the reusable partial function (the template) for the block
        block_partial = block_factory(specific_configs)

        # --- Build the sequence of blocks ---
        blocks = []
        for _ in range(num_blocks):
            # Instantiate the block from the template, enforcing dimension preservation.
            block = block_partial(
                in_channels=channels,
                out_channels=channels, # Channels are constant
                stride=stride              # Stride is always 1
            )
            blocks.append(block)
            
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)

    @staticmethod
    def get_conv(configs):
        """
        Factory method to parse the YAML.
        It validates the config and prepares the arguments for __init__.
        """
        configs = configs.copy()

        if "num_blocks" not in configs:
            raise ValueError("BlockSequence requires 'num_blocks' to be specified.")
        num_blocks = configs.pop("num_blocks")

        if "conv" not in configs:
            raise ValueError("BlockPreservingSequence requires a 'conv' key for the block template.")
        block_config = configs.pop("conv")

        if len(configs) > 0:
            raise ValueError(
                f"Unexpected keys in BlockSequence: {list(configs.keys())}"
            )
        
                # 3. Validate the extracted template to ensure it only defines one block type.
        if len(block_config) != 1:
            raise ValueError(f"The 'conv' section must define exactly one block type, but found {list(block_config.keys())}")

        def create_sequence_partial(in_channels, out_channels, stride):
            if in_channels != out_channels:
                raise ValueError(f"BlockSequence cannot change channels. Got in={in_channels}, out={out_channels}")
            if stride != 1:
                raise ValueError(f"BlockSequence cannot have stride != 1. Got stride={stride}")

            # The validated arguments are passed to the constructor.
            return BlockSequence(
                channels=in_channels,
                stride=stride,
                num_blocks=num_blocks,
                block_config=block_config
            )

        return create_sequence_partial
    

@register.NAME_TO_CONVS.register("TimeAwareBlockSequence")
class TimeAwareBlockSequence(BlockSequence):
    """
    A specialized framework class that generates a sequence of time-aware blocks.
    Its forward method correctly passes (t, x) to each block in the sequence.
    It is guaranteed to preserve the spatial size and channel count.
    """

    def forward(self, t, x):
        """
        A non-polymorphic, high-performance forward pass for time-aware modules.
        It expects two arguments, t and x, and passes them to its children.
        """
        # Sequentially apply each time-aware block.
        # The contract is that each block takes (t, x) and returns a new x.
        current_x = x
        for block in self.blocks:
            current_x = block(t, current_x)
        
        return current_x

    @staticmethod
    def get_conv(configs):
        """
        The factory method for this class. It's identical to the one for
        the standard BlockSequence, just pointing to its own class name.
        """
        configs = configs.copy()

        if "num_blocks" not in configs:
            raise ValueError("TimeAwareBlockSequence requires 'num_blocks'.")
        num_blocks = configs.pop("num_blocks")

        if "conv" not in configs:
            raise ValueError("TimeAwareBlockSequence requires a 'conv' key.")
        block_config = configs.pop("conv")
        
        if len(configs) > 0:
            raise ValueError(f"Unexpected keys in TimeAwareBlockSequence: {list(configs.keys())}")
        if len(block_config) != 1:
            raise ValueError(f"The 'conv' section must define one block type, but found {list(block_config.keys())}")

        def create_sequence_partial(in_channels, out_channels, stride):
            if in_channels != out_channels:
                raise ValueError(f"TimeAwareBlockSequence cannot change channels. Got in={in_channels}, out={out_channels}")
            if stride != 1:
                raise ValueError(f"TimeAwareBlockSequence cannot have stride != 1. Got stride={stride}")

            return TimeAwareBlockSequence(
                channels=in_channels,
                stride=stride,
                num_blocks=num_blocks,
                block_config=block_config
            )

        return create_sequence_partial