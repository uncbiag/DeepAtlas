#!/usr/bin/env python
"""
Created by zhenlinx on 12/20/2020
"""
import os
import sys
sys.path.append(os.path.realpath(".."))
import time
import datetime
import torch
import numpy as np
import random
from tqdm import tqdm, trange
from tensorboardX import SummaryWriter

import lib.utils as utils
from lib.loss import get_loss_function
import lib.transforms as med_transform
import lib.datasets as med_data
from lib.network_factory import get_network
import lib.visualize as vis
import lib.evalMetrics as metrics
from lib.param_dict import save_dict_to_json, load_jason_to_dict

class BaseExperiment():
    def __init__(self, config, **kwargs):
        self.config = config
        pass

    def setup_log(self):
        pass

    def setup_random_seed(self):
        torch.manual_seed(self.config['random_seed'])
        torch.cuda.manual_seed(self.config['random_seed'])
        np.random.seed(self.config['random_seed'])
        random.seed(self.config['random_seed'])
        # torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.deterministic = True

    def setup_train_data(self):
        pass

    def setup_model(self):
        pass

    def setup_loss(self):
        pass

    def setup_optimizer(self):
        pass

    def setup_train(self):
        self.setup_log()
        self.setup_random_seed()
        self.setup_model()
        self.setup_loss()
        self.setup_train_data()
        self.setup_optimizer()

    def train(self, **kwargs):
        raise NotImplementedError()

    def train_one_epoch(self, **kwargs):
        raise NotImplementedError()

    def validate(self, **kwargs):
        raise NotImplementedError()

    @staticmethod
    def save_checkpoint(state, is_best, path, prefix=None, name='checkpoint.pth.tar', max_keep=1):
        if not os.path.exists(path):
            os.makedirs(path)
        name = '_'.join([prefix, name]) if prefix else name
        best_name = '_'.join([prefix, 'model_best.pth.tar']) if prefix else 'model_best.pth.tar'
        torch.save(state, os.path.join(path, name))
        if is_best:
            torch.save(state, os.path.join(path, best_name))

    @staticmethod
    def initialize_model(model, optimizer=None, ckpoint_path=None):
        """
        Initilaize a reg_model with saved checkpoins, or random values
        :param model: a pytorch reg_model to be initialized
        :param optimizer: optional, optimizer whose parameters can be restored from saved checkpoints
        :param ckpoint_path: The path of saved checkpoint
        :return: currect epoch and best validation score
        """
        finished_epoch = 0
        best_score = 0
        if ckpoint_path:
            if os.path.isfile(ckpoint_path):
                print("=> loading checkpoint '{}'".format(ckpoint_path))
                checkpoint = torch.load(ckpoint_path, map_location=next(
                    model.parameters()).device)
                if 'best_score' in checkpoint:
                    best_score = checkpoint['best_score']
                elif 'reg_best_score' in checkpoint:
                    best_score = checkpoint['reg_best_score']
                elif 'seg_best_score' in checkpoint:
                    best_score = checkpoint['seg_best_score']
                else:
                    raise ValueError('no best score key')

                if type(best_score) is torch.Tensor:
                    best_score = best_score.cpu().item()

                model.load_state_dict(checkpoint['model_state_dict'], strict=True)
                if optimizer and checkpoint.__contains__('optimizer_state_dict'):
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                finished_epoch = finished_epoch + checkpoint['epoch']
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(ckpoint_path, checkpoint['epoch']))
                del checkpoint
            else:
                raise ValueError("=> no checkpoint found at '{}'".format(ckpoint_path))
        else:
            # reg_model.apply(weights_init)
            model.weights_init()
        return finished_epoch, best_score