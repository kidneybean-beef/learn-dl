import functools
import torch
import torch.nn as nn
import torch.nn.functional as F

import register
from classification import utils
from nn_module.conv.convs import Conv2d


@register.NAME_TO_CONVS.register("ODEBlock")
class ODEBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride, odefunc, solverfunc):
        super().__init__()
        if in_channels != out_channels:
            raise ValueError(
                f"ODEBlock must be used in a dimension-preserving stage, but in_channels ({in_channels}) "
                f"!= out_channels ({out_channels}). Check your network config."
            )
        if stride != 1:
            raise ValueError(
                f"ODEBlock must have a stride of 1, but got {stride}. Check your neiwork config."
            )
        self.odefunc = odefunc
        # This is now the pre-configured partial object from get_solver
        self.solver = solverfunc 
        self.integration_time = torch.tensor([0, 1]).float()

    def forward(self, x):
        self.integration_time = self.integration_time.type_as(x)
        
        # Simply call the solver. All options (method, tol, etc.) are already baked in.
        out = self.solver(func=self.odefunc, y0=x, t=self.integration_time)
        
        return out[1] # Return the solution at t=1
    
    @staticmethod
    def get_conv(configs):
        """
        Factory method for parsing the YAML and creating the ODEBlock.
        This now uses a wrapper to correctly handle arguments from the generic builder.
        """
        def create_ode_block_partial(in_channels, out_channels, stride):
            # 1. Get the ODEFunc configuration from the main configs
            # odefunc_config = configs["ODEFunc"]
            
            # 2. Get the partial function for the ODEFunc's main block
            #    (e.g., DimensionPreservingSequence)
            odefunc_builder_partial = register.get_conv(configs, "ODEFunc")

            # 3. Instantiate the ODEFunc's main block.
            #    Crucially, we pass the runtime in_channels, out_channels, and stride
            #    to THIS builder, so it can validate them.
            #    For DimensionPreservingSequence, it will check that channels are equal
            #    and stride is 1.
            odefunc_instance = odefunc_builder_partial(
                in_channels=in_channels, 
                out_channels=out_channels, 
                stride=stride
            )

            # 4. Get the solver function (this part is fine)
            solver_config = configs["solver"]
            solver_func = utils.get_solver(solver_config)
            
            # 5. Now, create the final ODEBlock instance
            return ODEBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                odefunc=odefunc_instance, # Pass the built module
                solverfunc=solver_func
            )
            
        return create_ode_block_partial

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value