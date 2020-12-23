#!/usr/bin/env python
"""
Created by zhenlinx on 11/4/18
"""
from . import modules
from . import voxel_morph
from . import unets

network_dic = {
    'voxel_morph_cvpr': voxel_morph.VoxelMorphCVPR2018,
    'UNet': unets.UNet,
    'UNet_light': unets.UNet_generator(encoders=[(8, 16), (16, 16, 32), (32, 32, 64), (64, 64, 64)],
                                       decoders=[(64, 64, 64), (64, 32, 32), (32, 16, 16)],
                                       act='LeakyReLU', maxpool=True, upsample=False, res=False
                                       )
}


def get_network(network_name):
    if network_name in get_available_networks():
        return network_dic[network_name]
    else:
        raise KeyError("Network \"{}\" is not avaiable!\n Choose from: {}".format(network_name, get_available_networks()))


def get_available_networks():
    return tuple(network_dic.keys())