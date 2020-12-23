#!/usr/bin/env python
"""
Created by zhenlinx on 11/14/18
"""
import os
import sys

sys.path.append(os.path.realpath(".."))

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import torch
import numpy as np
import scipy.misc
from skimage import color
import torchvision.utils as vision_utils


def plot_grad_flow(named_parameters, toFigure=False):
    """
    count and visualized avg grdient of each layer
    https://discuss.pytorch.org/t/check-gradient-flow-in-network/15063/5
    :param named_parameters: get from model.named_parameters if model is a nn.Module instance
    :return:
    """
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())

    if toFigure:
        fig = Figure(figsize=10,dpi=20)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis('equal')
        plt.plot(ave_grads, alpha=0.3, color="b")
        plt.hlines(0, 0, len(ave_grads) + 1, linewidth=1, color="k")
        # ax.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
        # ax.xlim(xmin=0, xmax=len(ave_grads))
        # ax.xlabel("Layers")
        # ax.ylabel("average gradient")
        # ax.title("Gradient flow")
        ax.grid(True)
        canvas = FigureCanvas(fig)
        canvas.draw()
        width, height = fig.get_size_inches() * fig.get_dpi()
        image = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3) / 255
        # image = np.transpose(image, [2, 0, 1])
        return image

    else:
        plt.figure()
        plt.plot(ave_grads, alpha=0.3, color="b")
        plt.hlines(0, 0, len(ave_grads) + 1, linewidth=1, color="k")
        plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
        plt.xlim(xmin=0, xmax=len(ave_grads))
        plt.xlabel("Layers")
        plt.ylabel("average gradient")
        plt.title("Gradient flow")
        plt.grid(True)
        plt.show()

def test_plot_gradient():
    x = torch.rand(3,3,10,10)
    truth = torch.rand(3,8,8,8)
    conv = torch.nn.Conv2d(3,8,3, padding=0, )
    y = conv(x)
    loss = ((y-truth)**2).mean()
    loss.backward()
    image = plot_grad_flow(conv.named_parameters(), True)
    scipy.misc.imsave('./gradient.jpg', image)


def new_plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    from matplotlib.lines import Line2D
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])

def generate_deform_grid(transform, slice_axis, background_image=None, n_bins=20):
    """
    Abandoned
    :param background_image: 1xMxN or 3xMxN tensor or numpy array
    :param transform: 3xMxN tensor or numpy array, the first axis are z,y,x coordinates
    :param slice_axis: which axis the slice is taken from a 3d volume,
    if 0, it is taken from z axis, than it is a x-y slice; Similarly, 1 for y, 2 for x
    :return: image numpy array MxNx3
    """
    if isinstance(transform, torch.Tensor):
        transform = transform.cpu().numpy()
    if background_image is not None:
        if isinstance(background_image, torch.Tensor):
            background_image = background_image.cpu().numpy()
        assert background_image.shape[1:] == transform.shape[1:]
    left_axis = [0, 1, 2]
    left_axis.remove(2-slice_axis)
    fig = Figure(figsize=np.array(transform.shape[1:])/5, dpi=20)
    ax = fig.add_axes([0, 0, 1, 1], frameon=False)
    # ax = fig.add_subplot(111)
    ax.set_axis_off()
    ax.axis('equal')
    xx = np.arange(0, transform.shape[1])
    yy = np.arange(0, transform.shape[2])
    if background_image is not None:
        ax.imshow(background_image.squeeze(), vmin=0, vmax=1, cmap='gray')

    # ax.set_ylim([0, background_image.shape[0]])

    for i, axis in enumerate(left_axis):
        T_slice = transform[axis, :, :]
        ax.contour(T_slice, colors=['yellow'], linewidths=10.0, linestyles='solid', levels=np.linspace(-1, 1, n_bins))
    ax.set_xlim([0, transform.shape[2]])
    plt.autoscale(tight=True)
    canvas = FigureCanvas(fig)
    canvas.draw()
    width, height = fig.get_size_inches() * fig.get_dpi()
    image = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)/255

    return np.transpose(image, [2,0,1])


def _generate_deform_grid(transform, slice_axis, background_image=None):
    """
    Abandoned
    :param transform: 3xMxN tensor, the first axis are z,y,x coordinates
    :param slice_axis: which axis the slice is taken from a 3d volume,
    if 0, it is taken from z axis, than it is a x-y slice; Similarly, 1 for y, 2 for x
    :return:
    """
    if isinstance(transform, torch.Tensor):
        transform = transform.cpu().numpy()
    if background_image is not None:
        if isinstance(background_image, torch.Tensor):
            background_image = background_image.cpu().numpy()
        assert background_image.shape[1:] == transform.shape[1:]

    left_axis = [0, 1, 2]
    left_axis.remove(2 - slice_axis)

    # if background_image is not None:
    #     # convert gray image to rgb image
    #     if background_image.shape[0] == 1:
    #         background_image = np.repeat(background_image, 3, axis=0)

    fig = plt.figure(figsize=np.array(transform.shape[1:]) / 5, dpi=10)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.axis('equal')
    # ax = fig.gca()
    if background_image is not None:
        ax.imshow(background_image.squeeze(), vmin=0, vmax=1, cmap='gray')
    xx = np.arange(0, transform.shape[1])
    yy = np.arange(0, transform.shape[2])
    for i, axis in enumerate(left_axis):
        T_slice = transform[axis, :, :]
        CSY = ax.contour(T_slice, colors=['yellow'], linewidths=10.0, linestyles = 'solid', levels=np.linspace(-1, 1, 10))
        # , levels = [-0.5, 0.5], colors=['white'])
    # ax.axis('off')
    # plt.draw()
    plt.show()
    # width, height = fig.get_size_inches() * fig.get_dpi()
    # image = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)

    # return image
    return None

def test_generate_deform_grid():
    from utils.tools import get_identity_transform_batch
    transform3d = get_identity_transform_batch([1, 1, 180, 160, 160])
    idslice = transform3d[0,:, 80, :, :]
    bk_img = torch.ones(1,180,160)/2
    image = _generate_deform_grid(idslice, 0, bk_img)
    # scipy.misc.imsave('grid.jpg', image)
    plt.figure()
    plt.imshow(image)
    plt.show()


def make_segmentation_image_summary(images, truths, raw_output, maxoutput=4, overlap=True, slice_ind=None, alpha=0.7):
    """make image summary for tensorboard

    :param images: torch.Variable, NxCxDxHxW, 3D image volume (C:channels)
    :param truths: torch.Variable, NxDxHxW, 3D label masks
    :param raw_output: torch.Variable, NxCxHxWxD: prediction for each class (C:classes)
    :param maxoutput: int, number of samples from a batch
    :param overlap: bool, overlap the image with groundtruth and predictions
    :return: summary_images: list, a maxoutput-long list with element of tensors of Nx
    """
    if not slice_ind:
        slice_ind = images.size()[2] // 2
    images_2D = images.data[:maxoutput, :, slice_ind, :, :]
    truths_2D = truths.data[:maxoutput, slice_ind, :, :]
    predictions_2D = torch.max(raw_output.data, 1)[1][:maxoutput, slice_ind, :, :]

    grid_images = vision_utils.make_grid(images_2D, pad_value=1)
    grid_truths = vision_utils.make_grid(labels2colors(truths_2D, images=images_2D, overlap=overlap, alpha=alpha), pad_value=1)
    grid_preds = vision_utils.make_grid(labels2colors(predictions_2D, images=images_2D, overlap=overlap, alpha=alpha), pad_value=1)

    return torch.cat([grid_images, grid_truths, grid_preds], 1)


def make_registration_image_summary(source_image, target_image, warped_source_image, disp_field, deform_field,
                                    source_seg=None, target_seg=None, warped_source_seg=None, n_slices=1, n_samples=1):
    """
    make image summary for tensorboard
    the image/seg grid are ordered in row by source, warped source, target and in column by HW, DW, DH slice
    the deform/disp field are ordered in row by [D, H, W]? value, target and in column by HW, DW, DH slice

    :param source_image: torch.tensor, NxCxDxHxW, 3D image volume (C:channels)
    :param target_image: torch.tensor, NxCxDxHxW, 3D image volume (C:channels)
    :param warped_source: torch.tensor, NxCxDxHxW, 3D image volume (C:channels)
    :param disp_field: torch.tensor, Nx3xDxHxW, 3D image volume, =deform_field -identity_transform
    :param deform_field: torch.tensor, Nx3xDxHxW, 3D image volume normalized in range [-1,1]
    :param n_slices: int, number of slices from a image volume
    :param n_samples: int, number of samples in a batch used from summary
    :param source_seg:
    :param warped_source_seg:
    :param target_seg:
    :return:
    """
    n_samples = min(n_samples, source_image.size()[0])
    grids = {}
    image_slices = []
    disp_slices = []
    seg_slices = []
    deform_grid_slices = []
    max_size = torch.tensor(source_image.shape[2:]).max().item()
    for n in range(n_samples):
        for axis in range(3):
            axis += 1
            # slice_ind = torch.arange(0, source_image.size()[axis], source_image.size()[axis + 2]/(n_slices+1))[1:]
            slice_ind = source_image.size()[axis + 1] // 2
            source_image_slice = torch.select(source_image[n, :, :, :, :], axis, slice_ind)
            warped_source_image_slice = torch.select(warped_source_image[n, :, :, :, :], axis, slice_ind)
            target_image_slice = torch.select(target_image[n, :, :, :, :], axis, slice_ind)
            image_slices += [source_image_slice, warped_source_image_slice, target_image_slice]

            disp_field_slice = torch.select(disp_field[n, :, :, :, :], axis, slice_ind)
            # disp_slices += [disp_field_slice[0:1, :, :], disp_field_slice[1:2, :, :],
            #                 disp_field_slice[2:3, :, :]]
            disp_slices+=[disp_field_slice]

            deform_field_slice = torch.select(deform_field[n, :, :, :, :], axis, slice_ind)

            deform_grid_slice = torch.from_numpy(
                generate_deform_grid(deform_field_slice, axis - 1, warped_source_image_slice))
            deform_grid_slices += [deform_grid_slice]

            if (source_seg is not None) and (target_seg is not None) and (warped_source_seg is not None):
                source_seg_slice = torch.select(source_seg[n, :, :, :], axis - 1, slice_ind)
                source_seg_slice = labels2colors(source_seg_slice, images=source_image_slice.squeeze(0), overlap=True)

                target_seg_slice = torch.select(target_seg[n, :, :, :], axis - 1, slice_ind)
                target_seg_slice = labels2colors(target_seg_slice, images=target_image_slice.squeeze(0), overlap=True)

                warped_source_seg_slice = torch.select(warped_source_seg[n, :, :, :], axis - 1, slice_ind)
                warped_source_seg_slice = labels2colors(warped_source_seg_slice,
                                                        images=warped_source_image_slice.squeeze(
                                                            0), overlap=True)

                seg_slices += [source_seg_slice, warped_source_seg_slice, target_seg_slice]

        grids['images'] = vision_utils.make_grid(slices_padding(image_slices), pad_value=1, nrow=3, normalize=True, range=(0, 1))
        if seg_slices:
            grids['masks'] = vision_utils.make_grid(slices_padding(seg_slices), pad_value=1, nrow=3)
        grids['disp_field'] = vision_utils.make_grid(slices_padding(disp_slices), pad_value=1, nrow=1, normalize=True, range=(-0.1, 0.1))
        grids['deform_grid'] = vision_utils.make_grid(slices_padding(deform_grid_slices), pad_value=1, nrow=1)
    return grids

def slices_padding(slice_list):
    """pad a list of slices to a square size"""
    max_size = torch.tensor([torch.tensor(slice.shape[1:]).max().item() for slice in slice_list]).max()
    return [slice_padding(slice, max_size) for slice in slice_list]


def slice_padding(slice, size):
    """padding a image into a square size """
    if slice.shape[1] != size or slice.shape[2] != size:
        slice_padded = torch.ones([slice.shape[0], size, size])
        padding_size = (size - slice.shape[1], size - slice.shape[2])
        slice_padded[:, padding_size[0]//2:slice.shape[1]+ padding_size[0]//2,
        padding_size[1] // 2:slice.shape[2] + padding_size[1] // 2] = slice
        return slice_padded
    else:
        return slice


def labels2colors(labels, images=None, overlap=False, alpha=0.7):
    """Turn label masks into color images
    :param labels: torch.tensor, BxMxN or MxN
    :param images: torch.tensor, BxMxN or BxMxNx3 or MxN or MxNx3
    :param overlap: bool
    :return: colors: torch.tensor, Bx3xMxN
    """
    images = images.clamp_(0, 1)
    if len(labels.size()) == 3:
        colors = []
        if overlap:
            if images is None:
                raise ValueError("Need background images when overlap is True")
            else:
                for i in range(images.size()[0]):
                    image = images.squeeze(dim=1)[i, :, :]
                    label = labels[i, :, :]
                    colors.append(color.label2rgb(label.cpu().numpy(), image.cpu().numpy(), bg_label=0, alpha=alpha))
        else:
            for i in range(images.size()[0]):
                label = labels[i, :, :]
                colors.append(color.label2rgb(label.numpy(), bg_label=0))

        return torch.Tensor(np.transpose(np.stack(colors, 0), (0, 3, 1, 2)))
    elif len(labels.size()) == 2:
        if overlap:
            if images is None:
                raise ValueError("Need background images when overlap is True")
            else:
                labelcolor = color.label2rgb(labels.cpu().numpy(), images.cpu().numpy(), bg_label=0, alpha=alpha)
        else:
            labelcolor = color.label2rgb(labels.numpy(), bg_label=0)

        return torch.Tensor(np.transpose(labelcolor, (2, 0, 1)))


if __name__ == '__main__':
    # test_generate_deform_grid()
    test_plot_gradient()