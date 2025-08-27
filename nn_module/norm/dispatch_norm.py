import torch.nn as nn

import register


class DispatchNorm(nn.Module):
    """
    A generic normalization layer dispatcher.
    
    Args:
        norm_class (nn.Module): The normalization layer class 
                                (e.g., nn.BatchNorm2d, nn.GroupNorm).
        **kwargs: Arguments to be passed to the norm_class constructor.
    """
    def __init__(self, norm_class, **kwargs):
        super().__init__()
        # The 'norm_class' is now the actual class, not a string.
        # The 'kwargs' dictionary contains all its required parameters.
        self.norm_layer = norm_class(**kwargs)

    def forward(self, x):
        return self.norm_layer(x)


@register.NAME_TO_NORMS.register("IdentityNorm")
class IdentityNorm(nn.Module):
    """
    Identity norm.
    This is generally used as "Identity Norm" in network for
    convenient implementation of "no normalization".
    """

    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, x):
        return x