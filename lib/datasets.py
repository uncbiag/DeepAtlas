#!/usr/bin/env python
"""
Created by zhenlinx on 11/4/18
"""
import SimpleITK as sitk
import os
import sys
import collections

import torch
from torch.utils.data import Dataset

sys.path.append(os.path.realpath(".."))


class SegDataSetOAIZIB(Dataset):
    """
    a dataset class to load medical image data in Nifti format using simpleITK
    """

    def __init__(self, txt_files, data_dir, with_seg=True, preload=False, pre_transform=None, running_transform=None,
                 n_samples=None, shuffle=False):
        """

        :param txt_files: txt file with lines of "image_file, segmentation_file" or a list of such files
        :param data_dir: the data root dir
        :param with_seg: if use segmentation
        :param preload: if preload image (with runining pre_transforms)
        :param pre_transform: transforms on data before running time. Preloading will save cost on pre_transform
        :param running_transform: Transformations on a sample for runtime data augmentation, e.g. random cropping
        :param n_samples: (int or a list of ints)
                if int, it means that only use first n_samples samples in the dataset.
                if list of ints, it is the sample indices which be used.
                All are used if it is None
        """
        self.n_samples = n_samples
        self.data_dir = data_dir
        self.with_seg = with_seg
        self.preload = preload
        self.pre_transform = pre_transform
        self.running_transform = running_transform
        self.shuffle = shuffle

        self.image_list, self.segmentation_list, self.name_list = self.read_image_segmentation_list(txt_files,
                                                                                                    self.data_dir,
                                                                                                    self.n_samples)

        if len(self.image_list) != len(self.segmentation_list):
            raise ValueError("The numbers of images and segmentations are different")

        if preload:
            print("Preloading data:")
            self.sample_list = self.preload_samples()

        self.length = len(self.image_list)
        if self.shuffle:
            self.shuffle_id = torch.randperm(self.length)

    def __len__(self):
        return self.length

    def __getitem__(self, id):
        if self.shuffle:
            id = self.shuffle_id[id]
        sample = self.get_sample(id)

        # TODO: implement with_seg option in the reading data stage
        return [item for item in sample.values()]

    def get_sample(self, id):
        if self.preload:
            sample = self.sample_list[id]
        else:
            image_file_name = self.image_list[id]
            segmentation_file_name = self.segmentation_list[id] if self.with_seg else None
            image_name = self.name_list[id]

            sample = self.load_sample(image_name, image_file_name, segmentation_file_name, self.pre_transform)

        if self.running_transform:
            sample = self.running_transform(sample)
        return sample

    def preload_samples(self):
        """
        Preload samples from list of image/segmentation filenames, preprocessing is done if there is any
        Return list of dictionaries of loaded images/segmentation (simpleITK image instances)
        :return:
        """

        sample_list = []

        for image_file_name, segmentation_file_name, name in zip(
                self.image_list, self.segmentation_list, self.name_list):
            sample_list.append(
                self.load_sample(name, image_file_name, segmentation_file_name if self.with_seg else None,
                                 self.pre_transform))

        return sample_list

    @staticmethod
    def load_sample(name, image_file_name, segmentation_file_name=None, pre_transform=None):
        """
        Load a segmentation data sample into a dictionary
        :param image_file_name:
        :param segmentation_file_name:
        :param name:
        :param pre_transform:
        :return:
        """
        if not os.path.exists(image_file_name):
            raise ValueError(image_file_name + ' not exist!')

        if segmentation_file_name and not os.path.exists(segmentation_file_name):
            raise ValueError(segmentation_file_name + ' not exist!')
        sample = {}
        image = sitk.ReadImage(image_file_name)
        sample['image'] = image
        if segmentation_file_name:
            seg = sitk.ReadImage(segmentation_file_name)
            sample['segmentation'] = seg

        sample['name'] = name
        if pre_transform:
            sample = pre_transform(sample)
        return sample

    @staticmethod
    def read_image_segmentation_list(text_files, data_root='', n_samples=None):
        """
        Read image filename list (and segmentation filename list) from a text file or a series of files
        :param n_samples: only use the first n samples in list if int, or index iterator
        :param text_files: name(s) of txt files
        :param data_root: root of image data
        :return: image_list: a list of image filenames
                segmentation_list: a list of segmentation filenames
                name_list: a list of scan name (parsed from image filenames)
        """
        image_list = []
        segmentation_list = []
        name_list = []
        if isinstance(text_files, str):
            text_files = [text_files]

        sample_count = 0
        for text_file in text_files:
            with open(text_file) as file:
                for line in file:
                    if isinstance(n_samples, collections.Sequence):
                        if sample_count not in n_samples:
                            sample_count += 1
                            continue
                    elif type(n_samples) is int:
                        if sample_count >= n_samples:
                            sample_count += 1
                            continue
                    elif n_samples is not None:
                        raise TypeError("n_samples should be None, or int, or a sequence of int bug get {}".format(type(n_samples)))

                    image_name = line.strip("\n")
                    name_list.append(image_name)
                    sample_count += 1
                    image_list.append(os.path.join(data_root, image_name + "_image.nii.gz"))
                    segmentation_list.append(os.path.join(data_root, image_name + "_masks.nii.gz"))

        return image_list, segmentation_list, name_list

    # @staticmethod
    # def read_image_segmentation(text_files, data_root=''):
    #     """
    #     Similar to read_image_segmentation_list() but instead of returning lists of filenames,
    #     it returns list of loaded images (simpleITK image instances)
    #     :param text_files:
    #     :param data_root:
    #     :return:
    #     """
    #
    #     image_list = []
    #     segmentation_list = []
    #     name_list = []
    #     if isinstance(text_files, str):
    #         text_files = [text_files]
    #
    #     for text_file in text_files:
    #         with open(text_file) as file:
    #             for line in file:
    #                 image_name = line.strip("\n")
    #                 name_list.append(image_name)
    #                 image_file_name = os.path.join(data_root, image_name + "_image.nii.gz")
    #                 segmentation_file_name = os.path.join(data_root, image_name + "_masks.nii.gz")
    #                 # check file existence
    #                 if not os.path.exists(image_file_name):
    #                     print(image_file_name + ' not exist!')
    #                     continue
    #                 if not os.path.exists(segmentation_file_name):
    #                     print(segmentation_file_name + ' not exist!')
    #                     continue
    #
    #                 image_list.append(sitk.ReadImage(image_file_name))
    #                 segmentation_list.append(sitk.ReadImage(segmentation_file_name))
    #
    #     return image_list, segmentation_list, name_list

class SegDataSetOASIS(SegDataSetOAIZIB):

    @staticmethod
    def read_image_segmentation_list(text_files, data_root='', n_samples=None):
        """
        Read image filename list (and segmentation filename list) from a text file or a series of files
        :param n_samples: only use the first n samples in list
        :param text_files: name(s) of txt files
        :param data_root: root of image data
        :return: image_list: a list of image filenames
                segmentation_list: a list of segmentation filenames
                name_list: a list of scan name (parsed from image filenames)
        """
        image_list = []
        segmentation_list = []
        name_list = []
        if isinstance(text_files, str):
            text_files = [text_files]

        sample_count = 0
        for text_file in text_files:
            with open(text_file) as file:
                for line in file:
                    if (n_samples is not None) and isinstance(n_samples, collections.Sequence) and (
                            sample_count not in n_samples) \
                            or ((type(n_samples) is int) and (sample_count >= n_samples)):
                        continue
                    image_name = line.strip("\n")
                    name_list.append(image_name)
                    sample_count += 1
                    image_list.append(os.path.join(data_root, image_name + "_image.nii.gz"))
                    segmentation_list.append(os.path.join(data_root, image_name + "_seg.nii.gz"))

        return image_list, segmentation_list, name_list


class SegDataSetBrains(SegDataSetOAIZIB):
    """
    seg dataset for brain images (LPBA40,CUMC12,IBSR18,MGH10)
    """
    @staticmethod
    def read_image_segmentation_list(text_files, data_root='', n_samples=None):
        """
        Read image filename list (and segmentation filename list) from a text file or a series of files
        :param n_samples: only use the first n samples in list
        :param text_files: name(s) of txt files
        :param data_root: root of image data
        :return: image_list: a list of image filenames
                segmentation_list: a list of segmentation filenames
                name_list: a list of scan name (parsed from image filenames)
        """
        image_list = []
        segmentation_list = []
        name_list = []
        if isinstance(text_files, str):
            text_files = [text_files]

        sample_count = 0
        for text_file in text_files:
            with open(text_file) as file:
                for line in file:
                    if isinstance(n_samples, collections.Sequence):
                        if sample_count not in n_samples:
                            sample_count += 1
                            continue
                    elif type(n_samples) is int:
                        if sample_count >= n_samples:
                            sample_count += 1
                            continue
                    elif n_samples is not None:
                        raise TypeError(
                            "n_samples should be None, or int, or a sequence of int bug get {}".format(type(n_samples)))
                    image_name = line.strip("\n")
                    name_list.append(image_name)
                    sample_count += 1
                    image_list.append(os.path.join(data_root, 'brain_affine_icbm_hist_matched', image_name + ".nii"))
                    segmentation_list.append(os.path.join(data_root, 'label_affine_icbm_reID', image_name + ".nii"))

        return image_list, segmentation_list, name_list


class SegDataSetMindBoggle(SegDataSetOAIZIB):
    """
    seg dataset for MindBoggle101
    """

    @staticmethod
    def read_image_segmentation_list(text_files, data_root='', n_samples=None):
        """
        Read image filename list (and segmentation filename list) from a text file or a series of files
        :param n_samples: only use the first n samples in list
        :param text_files: name(s) of txt files
        :param data_root: root of image data
        :return: image_list: a list of image filenames
                segmentation_list: a list of segmentation filenames
                name_list: a list of scan name (parsed from image filenames)
        """
        image_list = []
        segmentation_list = []
        name_list = []
        if isinstance(text_files, str):
            text_files = [text_files]

        sample_count = 0
        for text_file in text_files:
            with open(text_file) as file:
                for line in file:
                    if isinstance(n_samples, collections.Sequence):
                        if sample_count not in n_samples:
                            sample_count += 1
                            continue
                    elif type(n_samples) is int:
                        if sample_count >= n_samples:
                            sample_count += 1
                            continue
                    elif n_samples is not None:
                        raise TypeError(
                            "n_samples should be None, or int, or a sequence of int bug get {}".format(type(n_samples)))
                    image_name = line.strip("\n")
                    name_list.append(image_name)
                    sample_count += 1
                    image_list.append(os.path.join(data_root, 'image_in_MNI152_normalized', image_name + ".nii.gz"))
                    segmentation_list.append(os.path.join(data_root, 'label_31_reID_merged', image_name + ".nii.gz"))

        return image_list, segmentation_list, name_list


class RegDataSetBrains(SegDataSetBrains):
    """
    dataset for image registration
    """

    def __init__(self, txt_files, data_dir, with_seg=True, preload=False, pre_transform=None, running_transform=None,
                 n_samples=None, shuffle=False):
        super(RegDataSetBrains, self).__init__(txt_files, data_dir, with_seg, preload,
                                               pre_transform, running_transform,
                                               n_samples, shuffle)
        if self.shuffle:
            self.shuffle_id = torch.randperm(self.length * (self.length - 1))

    def __len__(self):
        return self.length * (self.length - 1)

    def __getitem__(self, id):
        if self.shuffle:
            id = self.shuffle_id[id]
        fixed_ind = id // (self.length - 1)
        moving_ind = id % (self.length - 1)
        if moving_ind >= fixed_ind:
            moving_ind += 1

        sample1 = self.get_sample(moving_ind)
        sample2 = self.get_sample(fixed_ind)

        # in order image, seg, name, seg_onehot(optional)
        return [item for item in sample1.values()], [item for item in sample2.values()]


class RegDataSetMindBoggle(SegDataSetMindBoggle):
    """
    dataset for image registration
    """

    def __init__(self, txt_files, data_dir, with_seg=True, preload=False, pre_transform=None, running_transform=None,
                 n_samples=None, shuffle=False):
        super(RegDataSetMindBoggle, self).__init__(txt_files, data_dir, with_seg, preload,
                                               pre_transform, running_transform,
                                               n_samples, shuffle)
        if self.shuffle:
            self.shuffle_id = torch.randperm(self.length * (self.length - 1))

    def __len__(self):
        return self.length * (self.length - 1)

    def __getitem__(self, id):
        if self.shuffle:
            id = self.shuffle_id[id]
        fixed_ind = id // (self.length - 1)
        moving_ind = id % (self.length - 1)
        if moving_ind >= fixed_ind:
            moving_ind += 1

        sample1 = self.get_sample(moving_ind)
        sample2 = self.get_sample(fixed_ind)

        # in order image, seg, name, seg_onehot(optional)
        return [item for item in sample1.values()], [item for item in sample2.values()]


class RegDataSetOASIS(SegDataSetOASIS):
    """
    dataset for image registration
    """

    def __init__(self, txt_files, data_dir, with_seg=True, preload=False, pre_transform=None, running_transform=None,
                 n_samples=None, shuffle=False):
        super(RegDataSetOASIS, self).__init__(txt_files, data_dir, with_seg, preload,
                                              pre_transform, running_transform,
                                              n_samples, shuffle)
        if self.shuffle:
            self.shuffle_id = torch.randperm(self.length * (self.length - 1))

    def __len__(self):
        return self.length * (self.length - 1)

    def __getitem__(self, id):
        if self.shuffle:
            id = self.shuffle_id[id]
        fixed_ind = id // (self.length - 1)
        moving_ind = id % (self.length - 1)
        if moving_ind >= fixed_ind:
            moving_ind += 1

        sample1 = self.get_sample(moving_ind)
        sample2 = self.get_sample(fixed_ind)

        # in order image, seg, name, seg_onehot(optional)
        return [item for item in sample1.values()], [item for item in sample2.values()]

class RegDataSetOAIZIB(SegDataSetOAIZIB):
    """
    dataset for image registration
    """

    def __init__(self, txt_files, data_dir, with_seg=True, preload=False, pre_transform=None, running_transform=None,
                 n_samples=None, shuffle=False):
        super(RegDataSetOAIZIB, self).__init__(txt_files, data_dir, with_seg, preload,
                                               pre_transform, running_transform,
                                               n_samples, shuffle)
        if self.shuffle:
            self.shuffle_id = torch.randperm(self.length * (self.length - 1))

    def __len__(self):
        return self.length * (self.length - 1)

    def __getitem__(self, id):
        if self.shuffle:
            id = self.shuffle_id[id]
        fixed_ind = id // (self.length - 1)
        moving_ind = id % (self.length - 1)
        if moving_ind >= fixed_ind:
            moving_ind += 1

        sample1 = self.get_sample(moving_ind)
        sample2 = self.get_sample(fixed_ind)

        # in order image, seg, name, seg_onehot(optional)
        return [item for item in sample1.values()], [item for item in sample2.values()]


def get_seg_dataset(data):
    if data == 'OAI':
        dataset = SegDataSetOAIZIB
    elif data == 'OASIS':
        dataset = SegDataSetOASIS
    elif data in ['LPBA40', 'CUMC12', 'IBSR18', 'MGH10']:
        dataset = SegDataSetBrains
    elif data == 'MindBoggle':
        dataset = SegDataSetMindBoggle
    else:
        raise ValueError("Wrong dataset given!")
    return dataset

def get_reg_dataset(data):
    if data == 'OAI':
        dataset = RegDataSetOAIZIB
    elif data == 'OASIS':
        dataset = RegDataSetOASIS
    elif data in ['LPBA40', 'CUMC12', 'IBSR18', 'MGH10']:
        dataset = RegDataSetBrains
    elif data == 'MindBoggle':
        dataset = RegDataSetMindBoggle
    else:
        raise ValueError("Wrong dataset given!")
    return dataset