#!/usr/bin/env python
"""
Created by zhenlinx on 12/20/2020
"""
from .base import *

import os
import time
import datetime
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from torchvision import transforms
from tensorboardX import SummaryWriter


class SegmentationExperiment(BaseExperiment):
    def __init__(self, config):
        super(SegmentationExperiment, self).__init__(config)

        if self.config['debug_mode']:
            print("Debug mode")
            self.config['print_batch_period'] = 2
            self.config['valid_epoch_period'] = 2

        self.exp_name = \
            'Seg_{}{}{}{}{}{}{}{}'.format(
                '{}{}{}_'.format(self.config['model'], '_bias' if self.config['model_settings']['bias'] else '',
                                 '_BN' if self.config['model_settings']['BN'] else ''),
                os.path.basename(self.config['data_dir']),
                '_{}samples'.format(self.config["num_samples"]),
                '_batch_{}'.format(self.config['batch_size']),
                '_{}epochs'.format(self.config['n_epochs']),
                '_{}_{}'.format(self.config['loss'], self.config['loss_settings']['weight_type']),
                '_lr_{}'.format(self.config['learning_rate']),
                '_scheduler_{}'.format(self.config['lr_mode']) if not self.config['lr_mode'] == 'const' else ''
            )

        self.ckpoint_dir = os.path.join(self.config['log_dir'],
                                        self.exp_name if not self.config['debug_mode'] else "debug_seg",
                                        str(self.config['random_seed']))
        print("Init experiment {} seed {}".format(self.exp_name, self.config['random_seed']))

    def setup_log(self):
        if not os.path.isdir(self.ckpoint_dir):
            os.makedirs(self.ckpoint_dir)
        save_dict_to_json(self.config, os.path.join(self.ckpoint_dir, "train_config.json"))

        # total_validation_time = 0
        # log writer
        self.writer = SummaryWriter(self.ckpoint_dir)

    def setup_train_data(self):
        print("Initializing dataloader")
        # set up data loader
        train_transforms = []
        train_transforms.append(med_transform.SitkToTensor())
        if self.config["crop_size"]:
            train_transforms.append(med_transform.CropTensor(self.config["crop_size"]))

        valid_transforms = train_transforms
        train_transforms = transforms.Compose(train_transforms)
        valid_transforms = transforms.Compose(valid_transforms)

        dataset = med_data.get_seg_dataset(self.config['data'])

        # assert (args.fold + 1) * self.config['num_samples'] <= self.config['max_samples']
        training_data = dataset(self.config['training_list_file'], self.config['data_dir'], with_seg=True,
                                preload=self.config['preload'], pre_transform=train_transforms,
                                n_samples=self.config["num_samples"] * 2)
        self.training_data_loader = DataLoader(training_data, batch_size=self.config['batch_size'],
                                               shuffle=True, num_workers=0 if self.config['debug_mode'] else 4)

        validation_data = dataset(self.config['validation_list_file'], self.config['valid_data_dir'], with_seg=True,
                                  preload=self.config['preload'], pre_transform=valid_transforms,
                                  )
        self.validation_data_loader = DataLoader(validation_data, batch_size=1,
                                                 shuffle=False, num_workers=0 if self.config['debug_mode'] else 2)

    def setup_model(self):
        # build segmentation model
        model_type = get_network(self.config['model'])
        self.model = model_type(**self.config['model_settings'])
        self.model.cuda()

    def setup_loss(self):
        self.criterion = get_loss_function(self.config['loss'])(**self.config['loss_settings']).cuda()

    def setup_optimizer(self):
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config['learning_rate'])

        # optimizer scheduler setting
        if self.config['lr_mode'] == 'plateau':
            plateau_threshold = 0.003
            plateau_patience = 100
            plateau_factor = 0.2
            self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max',
                                                            patience=plateau_patience // self.config[
                                                                'valid_epoch_period'],
                                                            factor=plateau_factor, verbose=True, threshold_mode='abs',
                                                            threshold=plateau_threshold,
                                                            min_lr=1e-5)

        elif self.config['lr_mode'] == 'multiStep':
            self.config['milestones'] = [int(ratio * self.config['n_epochs']) for ratio in self.config['milestones']]
            self.scheduler = lr_scheduler.MultiStepLR(self.optimizer, self.config['milestones'],
                                                      gamma=self.config['gamma'])

        else:
            self.scheduler = None

    def train(self):
        self.setup_train()
        print("Training {}".format(self.exp_name))

        # init/resume model
        finished_epoch, self.best_score = self.initialize_model(self.model, self.optimizer, self.config['resume_dir'])
        self.current_epoch = finished_epoch + 1

        print(self.config['samples_per_epoch'], self.config['batch_size'])

        print("Start Training:")
        for epoch in range(self.current_epoch, self.config['n_epochs'] + 1):
            self.train_one_epoch()
            self.validate()
            self.current_epoch += 1
        self.writer.close()
        print('Finished Training: {}'.format(self.exp_name))

    def train_one_epoch(self):
        running_loss = 0.0
        start_time = time.time()  # log running time

        iters_per_epoch = self.config['samples_per_epoch'] // self.config['batch_size']

        with trange(iters_per_epoch,
                    desc='Epoch[{}/{}] Train'.format(self.current_epoch, self.config['n_epochs'])) as t:
            for i in t:

                self.model.train()
                self.optimizer.zero_grad()
                try:
                    images, truths, name = next(train_data_iter)
                except:
                    train_data_iter = iter(self.training_data_loader)
                    images, truths, name = next(train_data_iter)

                self.global_step = (self.current_epoch - 1) * iters_per_epoch + (i + 1) * self.config[
                    'batch_size']  # current globel step

                output = self.model(images.cuda())

                loss = self.criterion(output, truths.long().cuda())

                loss.backward()
                self.optimizer.step()

                # print statistics
                running_loss += loss.item()  # average loss over batches
                if i % self.config['print_batch_period'] == self.config[
                    'print_batch_period'] - 1:

                    t.set_postfix_str('loss: {:.3f} lr:{} {}'.format(
                        running_loss / self.config['print_batch_period'] if i > 0 else running_loss,
                        self.optimizer.param_groups[0]['lr'],
                        datetime.datetime.now().strftime("%D %H:%M:%S")
                    ))
                    self.writer.add_scalar('loss/training', running_loss / self.config['print_batch_period'],
                                           global_step=self.global_step)
                    self.writer.add_scalar('learning_rate', self.optimizer.param_groups[0]['lr'],
                                           global_step=self.global_step)
                    running_loss = 0.0

            if self.current_epoch % self.config['save_ckpts_epoch_period'] == 0:
                image_summary = vis.make_segmentation_image_summary(images, truths, output.cpu())
                self.writer.add_image("training", image_summary, global_step=self.global_step)

    def eval(self, dataloader):
        with torch.no_grad():
            self.model.eval()
            dice_per_class = torch.tensor([0.] * (self.config["n_classes"] - 1))
            start_time = time.time()  # log running time
            data_iter = iter(dataloader)
            for j in trange(len(dataloader)):
                images, truths, name = next(data_iter)

                pred = self.model(images.cuda()).cpu()

                for c in range(1, self.config["n_classes"]):
                    dice_per_class[c - 1] += metrics.metricEval('dice',
                                                                torch.max(pred, 1)[1].squeeze().numpy() == c,
                                                                truths.numpy() == c,
                                                                num_labels=2)

            dice_per_class = dice_per_class / (j + 1)
            dice_avg = dice_per_class.mean()

            sample_for_vis = {'img': images, 'truth': truths, 'pred': pred}

        return dice_per_class, dice_avg, sample_for_vis

    def validate(self):
        # validation
        if self.current_epoch % self.config['valid_epoch_period'] == 0:
            start_time = time.time()
            dice_per_class, dice_avg, samples = self.eval(self.validation_data_loader)
            if self.config['lr_mode'] == 'plateau':
                self.scheduler.step(dice_avg)
            else:
                self.scheduler.step()
            is_best = False
            if dice_avg > self.best_score:
                is_best = True
                self.best_score = dice_avg

            self.writer.add_scalar('validation_{}/dice_avg'.format(self.config['data']), dice_avg,
                                   global_step=self.global_step)
            for c in range(self.config["n_classes"] - 1):
                self.writer.add_scalar(
                    'validation_{}/dice_{}'.format(self.config['data'], self.config["class_name"][c + 1]),
                    dice_per_class[c],
                    global_step=self.global_step)

            image_summary = vis.make_segmentation_image_summary(samples['img'], samples['truth'], samples['pred'])
            self.writer.add_image("validation", image_summary, global_step=self.global_step)

            print("Validation: Dice Avg: {:.4f} ".format(dice_avg) +
                  ' '.join(["Dice_{}:{:.3f}".format(self.config["class_name"][c + 1], dice_per_class[c]) for c in
                            range(self.config["n_classes"] - 1)]) +
                  " {:.3f} sec) {}".format(time.time() - start_time,
                                           datetime.datetime.now().strftime("%D %H:%M:%S")))

            if self.current_epoch % self.config['save_ckpts_epoch_period'] == 0:
                self.save_checkpoint({'epoch': self.current_epoch,
                                      'model_state_dict': self.model.state_dict(),
                                      'optimizer_state_dict': self.optimizer.state_dict(),
                                      'best_score': self.best_score},
                                     is_best, self.ckpoint_dir)

    def setup_test_data(self):
        test_transforms = [med_transform.SitkToTensor(), med_transform.CropTensor(self.config['crop_size'])]

        test_transform = transforms.Compose(test_transforms)

        seg_dataset = med_data.get_seg_dataset(self.config['data'])

        testing_data = seg_dataset(self.config['testing_list_file'], self.config['data_dir'], with_seg=True,
                                   preload=False,
                                   running_transform=test_transform)
        self.testing_data_loader = DataLoader(testing_data, batch_size=1, shuffle=False, num_workers=2)

    def test(self, best=True, if_log=True):
        import logging
        self.setup_model()
        ckpoint_file = os.path.join(self.ckpoint_dir, 'model_best.pth.tar' if best else 'checkpoint.pth.tar')
        last_epoch, best_score = self.initialize_model(self.model, optimizer=None,
                                                       ckpoint_path=ckpoint_file)

        self.setup_test_data()
        dice_per_class, dice_avg, samples = self.eval(self.testing_data_loader)
        if if_log:
            logging.basicConfig(filename=os.path.join(self.ckpoint_dir, 'test_log.txt'),
                                level=logging.DEBUG)
            logging.getLogger().addHandler(logging.StreamHandler())
            logging.info('\n' + "=" * 50 + '\n')
            logging.info('Testing Model: ' + ckpoint_file + '({} epochs)'.format(last_epoch) + '\n')
            logging.info('Test data: ' + self.config['data_dir'] + '\n')
            logging.info('Test list: ' + self.config['testing_list_file'] + '\n')
            logging.info('\n' + "-" * 50 + '\n')
            logging.info('Dice_avg: {}'.format(dice_avg))
            for c in range(self.config["n_classes"] - 1):
                logging.info("Dice_{}:{:.3f}".format(self.config["class_name"][c + 1], dice_per_class[c]))
            logging.info('\n' + "-" * 50 + '\n')
