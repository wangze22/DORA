'''
@author: Wang Ze
'''
import torch.nn as nn

from cim_layers.layers_all import *

nn_layers = (nn.Conv2d, nn.Linear)

qn_linear_layers = (l_qn_lsq.Linear_qn_lsq,
                    )

qn_conv_layers = (l_qn_lsq.Conv2d_qn_lsq,
                  )
qn_layers = qn_linear_layers + qn_conv_layers

cim_linear_layers = (
    l_144k.Linear_lsq_144k,
    l_adda_cim.Linear_lsq_adda_cim,
)

cim_conv_layers = (
    l_144k.Conv2d_lsq_144k,
    l_adda_cim.Conv2d_lsq_adda_cim,
)
cim_layers = cim_linear_layers + cim_conv_layers

adda_linear_layers = (

    l_adda_cim.Linear_lsq_adda_cim,

)
adda_conv_layers = (

    l_adda_cim.Conv2d_lsq_adda_cim,

)
adda_layers = adda_conv_layers + adda_linear_layers

chip_linear_layers = (
    l_144k.Linear_lsq_144k,

)
chip_conv_layers = (
    l_144k.Conv2d_lsq_144k,

)
chip_on_chip_layers = chip_conv_layers + chip_linear_layers

custom_linear_layers = qn_linear_layers + cim_linear_layers + adda_linear_layers + chip_linear_layers

custom_conv_layers = qn_conv_layers + cim_conv_layers + adda_conv_layers + chip_conv_layers

custom_layers = custom_linear_layers + custom_conv_layers

linear_layers = custom_linear_layers + (nn.Linear,)
conv_layers = custom_conv_layers + (nn.Conv2d,)
op_layers = linear_layers + conv_layers

digital_compute_layers = ['enhance_layer', 'enhance_branch']
