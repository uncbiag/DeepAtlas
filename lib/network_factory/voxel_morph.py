#!/usr/bin/env python
"""
registration network described in voxelmorphm
Created by zhenlinx on 11/8/18
"""

import os
import sys

sys.path.append(os.path.realpath("../.."))

import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.network_factory.modules import convBlock
from lib.utils import get_identity_transform_batch

class VoxelMorphCVPR2018(nn.Module):
    """
    unet architecture for voxelmorph models presented in the CVPR 2018 paper.
    You may need to modify this code (e.g., number of layers) to suit your project needs.

    :param input_channel: channels of input data (2 for a pair of images)
    :param output_channel: channels of output data (3 for 3D registration)
    :param enc_filters: list of encoder filters. values represent the number of filters of each layer
           e.g. (16, 32, 32, 32, 32)
    :param dec_filters: list of decoder filters.
    """
    def __init__(self, input_channel=2, output_channel=3, enc_filters=(16, 32, 32, 32, 32),
                 dec_filters=(32, 32, 32, 8, 8)):
        super(VoxelMorphCVPR2018, self).__init__()

        self.input_channel = input_channel
        self.output_channel = output_channel
        self.enc_filters = enc_filters
        self.dec_filters = dec_filters

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.upsampling = nn.Upsample(scale_factor=2, mode="trilinear")

        for i in range(len(enc_filters)):
            if i == 0:
                self.encoders.append(convBlock(input_channel, enc_filters[i], stride=1, bias=True))
            else:
                self.encoders.append(convBlock(enc_filters[i-1], enc_filters[i], stride=2, bias=True))

        for i in range(len(dec_filters)):
            if i == 0:
                self.decoders.append(convBlock(enc_filters[-1], dec_filters[i], stride=1, bias=True))
            elif i < 4:
                self.decoders.append(convBlock(dec_filters[i-1] if i == 4 else dec_filters[i - 1] + enc_filters[4-i],
                                            dec_filters[i], stride=1, bias=True))
            else:
                self.decoders.append(convBlock(dec_filters[i-1], dec_filters[i], stride=1, bias=True))

        self.flow = nn.Conv3d(dec_filters[-1] + enc_filters[0], output_channel, kernel_size=3, stride=1, padding=1, bias=True)

        # identity transform for computing displacement
        self.id_transform = None

    def forward(self, source, target):


        x_enc_1 = self.encoders[0](torch.cat((source, target), dim=1))
        # del input
        x_enc_2 = self.encoders[1](x_enc_1)
        x_enc_3 = self.encoders[2](x_enc_2)
        x_enc_4 = self.encoders[3](x_enc_3)
        x_enc_5 = self.encoders[4](x_enc_4)

        x_dec_1 = self.decoders[0](F.interpolate(x_enc_5, size=x_enc_4.shape[2:]))
        del x_enc_5
        x_dec_2 = self.decoders[1](F.interpolate(torch.cat((x_dec_1, x_enc_4), dim=1), size=x_enc_3.shape[2:]))
        del x_dec_1, x_enc_4
        x_dec_3 = self.decoders[2](F.interpolate(torch.cat((x_dec_2, x_enc_3), dim=1), size=x_enc_2.shape[2:]))
        del x_dec_2, x_enc_3
        x_dec_4 = self.decoders[3](torch.cat((x_dec_3, x_enc_2), dim=1))
        del x_dec_3, x_enc_2
        x_dec_5 = self.decoders[4](F.interpolate(x_dec_4, size=x_enc_1.shape[2:]))
        del x_dec_4
        disp_field = self.flow(torch.cat((x_dec_5, x_enc_1), dim=1))
        del x_dec_5, x_enc_1

        if self.id_transform is None:
            self.id_transform = get_identity_transform_batch(source.shape).to(disp_field.device)

        deform_field = disp_field + self.id_transform
        # transform images
        warped_source = F.grid_sample(source, grid=deform_field.permute([0,2,3,4,1]), mode='bilinear',
                                      padding_mode='zeros', align_corners=True)
        return disp_field, warped_source, deform_field

    def weights_init(self):
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                if not m.weight is None:
                    nn.init.xavier_normal_(m.weight.data)
                if not m.bias is None:
                    m.bias.data.zero_()


def demo_test():
    cuda = torch.device('cuda:1')

    # unet = UNet_light2(2,3).to(cuda)
    net = VoxelMorphCVPR2018().to(cuda)
    print(net)
    with torch.enable_grad():
        input1 = torch.randn(1, 1, 200, 200, 160).to(cuda)
        input2 = torch.randn(1, 1, 200, 200, 160).to(cuda)
        disp_field, warped_input1, deform_field = net(input1, input2)
    pass

if __name__ == '__main__':
    demo_test()