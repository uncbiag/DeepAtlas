#!/usr/bin/env python
"""
Created by zhenlinx on 11/10/19
"""
import torch

def get_identity_transform_batch(size, normalize=True):
    """
    generate an identity transform for given image size (NxCxDxHxW)
    :param size: Batch, D,H,W size
    :param normalize: normalized index into [-1,1]
    :return: identity transform with size Nx3xDxHxW
    """
    _identity = get_identity_transform(size[2:], normalize)
    return _identity


def get_identity_transform(size, normalize=True):
    """

    :param size: D,H,W size
    :param normalize:
    :return: 3XDxHxW tensor
    """

    if normalize:
        xx, yy, zz = torch.meshgrid([torch.arange(0, size[k]).float() / (size[k] - 1) * 2.0 - 1 for k in [0, 1, 2]])
    else:
        xx, yy, zz = torch.meshgrid([torch.arange(0, size[k]) for k in [0, 1, 2]])
    _identity = torch.stack([zz, yy, xx])
    return _identity
