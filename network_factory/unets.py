import torch
import torch.nn as nn
from collections import OrderedDict

available_activations = {'ReLU': nn.ReLU,
                         'LeakyReLU': nn.LeakyReLU}


# def get_activation_function(act):
#     """
#     Get an activation function
#     :param act:
#     :return:
#     """
#     if act in available_activations:
#         return  available_activations[act]
#     else:
#         NotImplementedError(
#             "Not Implemented activation type {}, only {} are available now".format(act, available_activations.keys()))
#
#
# # 3D UNet and its variants
#
# def convBlock(in_channels, out_channels, kernel_size=3, stride=1, padding=1,
#               bias=True, batchnorm=False, act='ReLU'):
#     act_F = get_activation_function(act)
#
#     if batchnorm:
#         layer = nn.Sequential(OrderedDict({
#             'conv': nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
#             'BN': nn.BatchNorm3d(out_channels),
#             'nonlinear': act_F()
#         }))
#     else:
#         layer = nn.Sequential(OrderedDict({
#             'conv': nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
#             'nonlinear': act_F()
#         }))
#     return layer
#
#
# def deconvBlock(in_channels, out_channels, kernel_size, stride=1, padding=0,
#                 output_padding=0, bias=True, batchnorm=False, act='ReLU'):
#     # activation layer
#     act_F = get_activation_function(act)
#
#     if batchnorm:
#         layer = nn.Sequential(OrderedDict({
#             'deconv': nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride,
#                                padding=padding, output_padding=output_padding, bias=bias),
#             'BN': nn.BatchNorm3d(out_channels),
#             'nonlinear': act_F()}))
#     else:
#         layer = nn.Sequential(OrderedDict({
#             'deconv': nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride,
#                                          padding=padding, output_padding=output_padding, bias=bias),
#             'nonlinear': act_F()}))
#     return layer


def init_conv_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        if not m.weight is None:
            nn.init.xavier_normal_(m.weight.data)
        if not m.bias is None:
            m.bias.data.zero_()


class UNet(nn.Module):
    def __init__(self, in_channel, n_classes, bias=False, BN=False):
        self.in_channel = in_channel
        self.n_classes = n_classes
        super(UNet, self).__init__()
        self.ec0 = self.encoder(self.in_channel, 32, bias=bias, batchnorm=BN)
        self.ec1 = self.encoder(32, 64, bias=bias, batchnorm=BN)
        self.ec2 = self.encoder(64, 64, bias=bias, batchnorm=BN)
        self.ec3 = self.encoder(64, 128, bias=bias, batchnorm=BN)
        self.ec4 = self.encoder(128, 128, bias=bias, batchnorm=BN)
        self.ec5 = self.encoder(128, 256, bias=bias, batchnorm=BN)
        self.ec6 = self.encoder(256, 256, bias=bias, batchnorm=BN)
        self.ec7 = self.encoder(256, 512, bias=bias, batchnorm=BN)

        self.pool0 = nn.MaxPool3d(2)
        self.pool1 = nn.MaxPool3d(2)
        self.pool2 = nn.MaxPool3d(2)

        self.dc9 = self.decoder(512, 512, kernel_size=2, stride=2, bias=bias, batchnorm=BN)
        self.dc8 = self.decoder(256 + 512, 256, kernel_size=3, stride=1, padding=1, bias=bias, batchnorm=BN)
        self.dc7 = self.decoder(256, 256, kernel_size=3, stride=1, padding=1, bias=bias, batchnorm=BN)
        self.dc6 = self.decoder(256, 256, kernel_size=2, stride=2, bias=bias, batchnorm=BN)
        self.dc5 = self.decoder(128 + 256, 128, kernel_size=3, stride=1, padding=1, bias=bias, batchnorm=BN)
        self.dc4 = self.decoder(128, 128, kernel_size=3, stride=1, padding=1, bias=bias, batchnorm=BN)
        self.dc3 = self.decoder(128, 128, kernel_size=2, stride=2, bias=bias, batchnorm=BN)
        self.dc2 = self.decoder(64 + 128, 64, kernel_size=3, stride=1, padding=1, bias=bias, batchnorm=BN)
        self.dc1 = self.decoder(64, 64, kernel_size=3, stride=1, padding=1, bias=bias, batchnorm=BN)
        # self.dc2_2 = self.decoder(64, n_classes, kernel_size=1, stride=1, bias=False)
        self.dc0 = nn.Conv3d(64, n_classes, kernel_size=1, stride=1, padding=0, bias=bias)

        # self.weights_init()

    def weights_init(self):
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                if not m.weight is None:
                    nn.init.xavier_normal_(m.weight.data)
                if not m.bias is None:
                    m.bias.data.zero_()

    def encoder(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                bias=True, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm3d(out_channels),
                nn.ReLU())
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.ReLU())
        return layer

    def decoder(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                output_padding=0, bias=True, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride,
                                   padding=padding, output_padding=output_padding, bias=bias),
                nn.BatchNorm3d(out_channels),
                nn.ReLU())
        else:
            layer = nn.Sequential(
                nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride,
                                   padding=padding, output_padding=output_padding, bias=bias),
                nn.ReLU())
        return layer

    def forward(self, x):
        e0 = self.ec0(x)
        syn0 = self.ec1(e0)
        e1 = self.pool0(syn0)
        e2 = self.ec2(e1)
        syn1 = self.ec3(e2)
        del e0, e1, e2

        e3 = self.pool1(syn1)
        e4 = self.ec4(e3)
        syn2 = self.ec5(e4)
        del e3, e4

        e5 = self.pool2(syn2)
        e6 = self.ec6(e5)
        e7 = self.ec7(e6)
        del e5, e6

        d9 = torch.cat((self.dc9(e7), syn2), dim=1)
        del e7, syn2

        d8 = self.dc8(d9)
        d7 = self.dc7(d8)
        del d9, d8

        d6 = torch.cat((self.dc6(d7), syn1), dim=1)
        del d7, syn1

        d5 = self.dc5(d6)
        d4 = self.dc4(d5)
        del d6, d5

        d3 = torch.cat((self.dc3(d4), syn0), dim=1)
        del d4, syn0

        d2 = self.dc2(d3)
        d1 = self.dc1(d2)
        del d3, d2

        d0 = self.dc0(d1)
        return d0


def UNet_generator(encoders, decoders, act='ReLU',
                   upsample=False,
                   maxpool=True,
                   res=False):
    """

    :param encoders:
    :param decoders:
    :param act: activation type
    :param upsample: if True, use upsample instead of deconvolution
    :param maxpool: if False, use strided convolution instead of maxpooling
    :param res: if True, use residual connection for each encoder level
    :return:
    """


    class UNetTemplate(nn.Module):
        def __init__(self, in_channel, n_classes,
                     bias=False, BN=False):
            """

            :param in_channel:
            :param n_classes:

            :param bias:
            :param BN:
            """
            super(UNetTemplate, self).__init__()
            self.in_channel = in_channel
            self.n_classes = n_classes
            self.levels = len(encoders)  # resolution level
            self.encoders = nn.ModuleList()
            self.decoders = nn.ModuleList()
            self.down_samplers = nn.ModuleList()
            self.up_samplers = nn.ModuleList()
            self.maxpool = maxpool
            self.upsample = upsample
            self.res = res

            # encoders[0] = (in_channel,) + encoders[0] this makes bugs !
            for i, enc in enumerate(encoders):
                if i == 0:
                    enc = (in_channel,) + enc
                temp_encBlocks = [convBlock(enc[k], enc[k + 1], bias=bias, batchnorm=BN, act=act) for k in
                                  range(len(enc) - 1)]
                self.encoders.append(nn.Sequential(*temp_encBlocks))

                if i < len(encoders) - 1:
                    self.down_samplers.append(nn.MaxPool3d(2) if self.maxpool
                                              else nn.Conv3d(enc[-1], encoders[i + 1][0], kernel_size=2, stride=2, padding=0,
                                                             bias=bias))

            for i, dec in enumerate(decoders):
                # upsampling
                if self.upsample:
                    self.up_samplers.append(nn.Upsample(scale_factor=2, mode="trilinear"))
                    pass
                else:
                    self.up_samplers.append(deconvBlock(encoders[-1][-1] if i == 0 else decoders[i - 1][-1], dec[0],
                                                        kernel_size=2, stride=2, bias=bias, batchnorm=BN, act=act))

                # conv layes in the same level
                dec = (encoders[-(i + 2)][-1] + dec[0],) + dec[1:]  # concatenation with encoder features
                temp_decBlocks = [convBlock(dec[k], dec[k+1], kernel_size=3, stride=1, padding=1, bias=bias, batchnorm=BN,
                                            act=act)
                                  for k in range(len(enc) - 1)]
                # output layer
                if i == len(decoders)-1:
                    temp_decBlocks.append(nn.Conv3d(dec[-1], n_classes, kernel_size=1, stride=1, padding=0, bias=bias))

                self.decoders.add_module('decBlock{}'.format(i), nn.Sequential(*temp_decBlocks))

            # self.weights_init()

        def weights_init(self):
            self.apply(init_conv_weights)

        def forward(self, x):
            temp = [None] * self.levels

            # encoders
            for i, enc in enumerate(self.encoders):
                temp[i] = (enc(x) + x) if self.res else enc(x)

                if i < self.levels - 1:
                    x = self.down_samplers[i](temp[i])
                else:
                    x = temp[i]
                    temp.pop()

            # decoder
            for j, dec in enumerate(self.decoders):
                x = self.up_samplers[j](x)
                x = dec(torch.cat((x, temp[-1]), dim=1)) + x if self.res else dec(torch.cat((x, temp[-1]), dim=1))
                temp.pop()

            return x

    return UNetTemplate
