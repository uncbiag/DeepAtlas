#!/usr/bin/env python
"""
modules for building networks
Created by zhenlinx on 11/4/18
"""
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F

available_activations = {'ReLU': nn.ReLU,
                         'LeakyReLU': nn.LeakyReLU}


def get_activation_function(act):
    """
    Get an activation function
    :param act:
    :return:
    """
    if act in available_activations:
        return available_activations[act]
    else:
        NotImplementedError(
            "Not Implemented activation type {}, only {} are available now".format(act, available_activations.keys()))


class convBlock(nn.Module):
    """
    A convolutional block including conv, BN, nonliear activiation, residual connection
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                 bias=False, batchnorm=False, act=nn.ReLU, residual=False, ):
        """

        :param in_channels:
        :param out_channels:
        :param kernel_size:
        :param stride:
        :param padding:
        :param bias:
        :param batchnorm:
        :param residual:
        :param act:
        """

        super(convBlock, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm3d(out_channels) if batchnorm else None
        self.nonlinear = get_activation_function(act) if type(act) is str else act
        self.residual = residual

    def forward(self, x):
        x= self.conv(x)
        if self.bn:
            x = self.bn(x)
        if self.nonlinear:
            x = self.nonlinear()(x)
        if self.residual:
            x += x

        return x


class deconvBlock(nn.Module):
    """
    A convolutional block including conv, BN, nonliear activiation, residual connection
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, output_padding=0, bias=False, batchnorm=False, residual=False, act=nn.ReLU):
        super(deconvBlock, self).__init__()
        self.deconv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride,
                              padding=padding, output_padding=output_padding, bias=bias)
        self.bn = nn.BatchNorm3d(out_channels) if batchnorm else None
        self.nonlinear = act
        self.residual = residual

    def forward(self, input):
        x = self.deconv(input)
        if self.bn:
            x = self.bn(x)
        x = self.nonlinear(x)
        if self.residual:
            x += input
        return x




def test_conv():
    input = torch.randn(20, 16, 10, 50, 100)
    conv = convBlock(16, 33, stride=2)
    deconv = deconvBlock(16,33, 3, stride=1)
    output = conv(input)
    deoutput = deconv(input)
    print(input.shape, output.shape)

if __name__ == '__main__':
    test_conv()