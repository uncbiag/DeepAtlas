#!/usr/bin/env python
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
import torch.nn.functional as F


class RegistrationExperiment(BaseExperiment):
    def __init__(self, config):
        super(RegistrationExperiment, self).__init__(config)

        if self.config['debug_mode']:
            print("Debug mode")
            self.config['print_batch_period'] = 2
            self.config['valid_epoch_period'] = 2

        self.weakly_supervised = False  # if we are using segmentation for training

        self.exp_name = 'Reg_{}{}{}{}{}{}{}{}{}'.format(
            self.config['reg_model'],
            '_t-{}'.format(
                os.path.basename(self.config['training_list_file'][0]).split('.')[0]),
            '_v-{}'.format(os.path.basename(self.config['validation_list_file']).split('.')[0]),
            '_batch_{}'.format(self.config['batch_size']),
            '_Simloss_' + '_'.join(
                '{}_{}'.format(name, weight) for (name, settings, weight) in self.config['sim_loss']),
            '_Segloss_' + '_'.join(
                '{}_{}_{}'.format(name, settings['weight_type'], weight) for (name, settings, weight) in
                self.config['seg_loss']),
            '_Regloss_' + '_'.join('{}_{}'.format(name, weight) for (name, settings, weight) in self.config['reg_loss'])
            if self.config['reg_loss'] != [] else '',
            '_lr_{}'.format(self.config['learning_rate']),
            '_scheduler_{}'.format(self.config['lr_mode']) if not self.config['lr_mode'] == 'const' else ''
        )

        self.ckpoint_dir = os.path.join(self.config['log_dir'],
                                        self.exp_name if not self.config['debug_mode'] else "debug_reg",
                                        str(self.config['random_seed']))
        print("Init experiment {} seed {}".format(self.exp_name, self.config['random_seed']))

    def setup_log(self):
        if not os.path.isdir(self.ckpoint_dir):
            os.makedirs(self.ckpoint_dir)
        save_dict_to_json(self.config, os.path.join(self.ckpoint_dir, "train_config.json"))

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

        dataset = med_data.get_reg_dataset(self.config['data'])

        training_data = dataset(self.config['training_list_file'], self.config['data_dir'],
                                with_seg=self.weakly_supervised,
                                preload=self.config['preload'],
                                pre_transform=train_transforms,
                                running_transform=med_transform.SegMaskToOneHot(self.config["n_classes"],
                                                                                dtype=torch.uint8)
                                if self.weakly_supervised else None,
                                n_samples=self.config['num_training_samples']
                                )

        validation_data = dataset(self.config['validation_list_file'], self.config['data_dir'],
                                  with_seg=True,
                                  preload=self.config['preload'],
                                  pre_transform=valid_transforms,
                                  running_transform=med_transform.SegMaskToOneHot(self.config["n_classes"],
                                                                                  dtype=torch.uint8)
                                  )

        self.training_data_loader = DataLoader(training_data, batch_size=self.config['batch_size'], shuffle=True,
                                               num_workers=0 if self.config['debug_mode'] else 4)

        self.validation_data_loader = DataLoader(validation_data, batch_size=self.config['valid_batch_size'],
                                                 shuffle=False, num_workers=0 if self.config['debug_mode'] else 4)

    def setup_model(self):
        # build segmentation model
        model_type = get_network(self.config['reg_model'])
        self.model = model_type(**self.config['reg_model_setting'])
        self.model.cuda()

    def setup_loss(self):
        self.sim_criterions = [(name, get_loss_function(name)(**setting).cuda(), weight) for
                               (name, setting, weight) in self.config['sim_loss'] if weight > 0]
        self.seg_criterions = [(name, get_loss_function(name)(**setting).cuda(), weight) for
                               (name, setting, weight) in self.config['seg_loss'] if weight > 0]
        self.reg_criterions = [(name, get_loss_function(name)(**setting).cuda(), weight) for
                               (name, setting, weight) in self.config['reg_loss'] if weight > 0]

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
            self.scheduler = lr_scheduler.MultiStepLR(self.optimizer, self.config['milestones'], gamma=0.5)
        else:
            self.scheduler = None

    def train(self):
        self.setup_train()
        print("Training {}".format(self.exp_name))

        # init/resume model
        finished_epoch, self.best_score = self.initialize_model(self.model, self.optimizer, self.config['resume_dir'])
        self.current_epoch = finished_epoch + 1


        print("Start Training:")
        for epoch in range(self.current_epoch, self.config['n_epochs'] + 1):
            self.train_one_epoch()
            self.validate()
            self.current_epoch += 1
        self.writer.close()
        print('Finished Training: {}'.format(self.exp_name))

    def train_one_epoch(self):
        running_losses = [['Total', 0.0]]
        running_losses += [[name, 0.0]
                           for name, _, _ in self.sim_criterions]
        if self.weakly_supervised:
            running_losses += [[name, 0.0]
                               for name, _, _ in self.seg_criterions]
        running_losses += [[name, 0.0]
                           for name, _, _ in self.reg_criterions]

        start_time = time.time()  # log running time
        batches_per_epoch = self.config['num_training_samples'] // self.config['batch_size'] \
            if self.config['num_training_samples'] is not None \
            else len(self.training_data_loader.dataset) // self.config['batch_size']

        if not self.config['lr_mode'] == 'const' and not self.config['lr_mode'] == 'plateau':
            self.scheduler.step(epoch=self.current_epoch)

        for i, (source_data, target_data) in enumerate(self.training_data_loader):
            if i > batches_per_epoch:
                break
            if self.weakly_supervised:
                (source_image, source_seg, name_source, source_seg_onehot) = source_data
                (target_image, target_seg, target_name, target_seg_onehot) = target_data
            else:
                (source_image, name_source) = source_data
                (target_image, target_name) = target_data

            self.model.train()

            self.global_step = (self.current_epoch - 1) * batches_per_epoch * self.config['batch_size'] \
                               + (i + 1) * self.config['batch_size']  # current globel step

            self.optimizer.zero_grad()

            source_image_on_device = source_image.cuda()
            target_image_on_device = target_image.cuda()

            disp_field, warped_source_image, deform_field = self.model(source_image_on_device, target_image_on_device)

            if self.weakly_supervised:
                warped_source_seg_onehot = F.grid_sample(source_seg_onehot.cuda().float(),
                                                         grid=deform_field.permute([0, 2, 3, 4, 1]),
                                                         mode='bilinear',
                                                         padding_mode='zeros')

            # image similarity loss
            losses = []
            if self.sim_criterions:
                losses += [(name, sim_criterion(target_image_on_device, warped_source_image), weight)
                           for name, sim_criterion, weight in self.sim_criterions]

            # segmentation overlapping loss
            if self.weakly_supervised:
                losses += [
                    (name, seg_criterion(warped_source_seg_onehot, target_seg_onehot.cuda().float()), weight)
                    for name, seg_criterion, weight in self.seg_criterions]

            # displacement field regularization loss
            losses += [(name, reg_criterion(disp_field), weight)
                       for name, reg_criterion, weight in self.reg_criterions]

            total_loss = 0
            for name, subloss, weight in losses:
                total_loss += subloss * weight

            total_loss.backward()
            self.optimizer.step()

            # print statistics
            running_losses[0][1] += total_loss.item()
            for k in range(1, len(running_losses)):
                running_losses[k][1] += losses[k - 1][1].item()

            # print
            if (i + 1) % self.config['print_batch_period'] == 0 or i == 0:
                duration = time.time() - start_time
                for k in range(len(running_losses)):
                    if i != 0:
                        running_losses[k][1] /= self.config['print_batch_period']

                print(
                    'Epoch: {:0d} [{}/{} ({:.0f}%)] {} lr:{} ({:.3f} sec/batch) {}'.format
                    (self.current_epoch,
                     (i + 1) * self.config['batch_size'], batches_per_epoch * self.config['batch_size'],
                     (i + 1) * self.config['batch_size'] / (batches_per_epoch * self.config['batch_size']) * 100,
                     ' '.join(['{}_loss: {:.3e}'.format(name, value) for name, value in running_losses]),
                     self.optimizer.param_groups[0]['lr'],
                     duration / self.config['print_batch_period'],
                     datetime.datetime.now().strftime("%D %H:%M:%S")
                     ))
                for k in range(len(running_losses)):
                    self.writer.add_scalar('training/{}_loss'.format(running_losses[k][0]), running_losses[k][1],
                                           global_step=self.global_step)

                self.writer.add_scalar('learning_rate', self.optimizer.param_groups[0]['lr'],
                                       global_step=self.global_step)
                for k in range(len(running_losses)):
                    running_losses[k][1] = 0
                start_time = time.time()

    def eval(self, dataloader):
        running_losses = [['Total', 0.0]]
        running_losses += [[name, 0.0]
                           for name, _, _ in self.sim_criterions]
        if self.weakly_supervised:
            running_losses += [[name, 0.0]
                               for name, _, _ in self.seg_criterions]
        running_losses += [[name, 0.0]
                           for name, _, _ in self.reg_criterions]

        with torch.no_grad():
            self.model.eval()
            dice_per_class = torch.zeros(self.config["n_classes"] - 1)  # no background class 0
            self.test_data_iter = iter(dataloader)

            for j in range(min(self.config['valid_samples'] // self.config['valid_batch_size'], len(
                    dataloader))):
                ((source_image, source_seg, source_name, source_seg_onehot),
                 (target_image, target_seg, target_name, target_seg_onehot)) = next(self.test_data_iter)
                source_image_on_device = source_image.cuda()
                target_image_on_device = target_image.cuda()
                disp_field, warped_source_image, deform_field = self.model(source_image_on_device,
                                                                           target_image_on_device)

                warped_source_seg_onehot = F.grid_sample(source_seg_onehot.cuda().float(),
                                                         grid=deform_field.permute([0, 2, 3, 4, 1]), mode='bilinear',
                                                         padding_mode='zeros')

                # compute validation losses
                losses = []
                if self.sim_criterions:
                    losses += [(name, sim_criterion(target_image_on_device, warped_source_image), weight)
                               for name, sim_criterion, weight in self.sim_criterions]

                # segmentation overlapping loss
                if self.seg_criterions:
                    losses += [
                        (name, seg_criterion(warped_source_seg_onehot, target_seg_onehot.cuda().float()),
                         weight)
                        for name, seg_criterion, weight in self.seg_criterions]

                # displacement field regularization loss
                losses += [(name, reg_criterion(disp_field), weight)
                           for name, reg_criterion, weight in self.reg_criterions]

                total_loss = 0
                for name, subloss, weight in losses:
                    total_loss += subloss * weight

                # accumulating losses over batches
                running_losses[0][1] += total_loss.item()
                for k in range(1, len(running_losses)):
                    running_losses[k][1] += losses[k - 1][1].item()

                # accumulating dice score in each batch
                for b in range(self.config['valid_batch_size']):
                    for c in range(1, self.config["n_classes"]):
                        dice_per_class[c - 1] += metrics.metricEval('dice',
                                                                    torch.argmax(
                                                                        warped_source_seg_onehot[b:b + 1, :, :, :,
                                                                        :].detach(), 1,
                                                                        False).cpu().squeeze().numpy() == c,
                                                                    target_seg[b:b + 1, :, :, :].numpy() == c,
                                                                    num_labels=2)
            # average dice score
            dice_per_class = dice_per_class / (j + 1)
            dice_avg = dice_per_class.mean()

        return dice_per_class, dice_avg

    def validate(self):
        # validation
        if self.current_epoch % self.config['valid_epoch_period'] == 0:
            start_time = time.time()
            dice_per_class, dice_avg = self.eval(self.validation_data_loader)
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
                    'validation_{}/dice_{}'.format(self.config['data'], self.config["class_name"][c]),
                    dice_per_class[c],
                    global_step=self.global_step)

            print("Validation: Dice Avg: {:.4f} ".format(dice_avg) +
                  ' '.join(["Dice_{}:{:.3f}".format(self.config["class_name"][c], dice_per_class[c]) for c in
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
        test_transforms = []
        test_transforms.append(med_transform.SitkToTensor())
        if self.config["crop_size"]:
            test_transforms.append(med_transform.CropTensor(self.config["crop_size"]))

        test_transforms = transforms.Compose(test_transforms)

        dataset = med_data.get_reg_dataset(self.config['data'])

        testing_data = dataset(self.config['testing_list_file'], self.config['data_dir'],
                               with_seg=True,
                               preload=self.config['preload'],
                               pre_transform=test_transforms,
                               running_transform=med_transform.SegMaskToOneHot(self.config["n_classes"],
                                                                               dtype=torch.uint8)
                               )

        self.testing_data_loader = DataLoader(testing_data, batch_size=self.config['batch_size'], shuffle=False,
                                              num_workers=0 if self.config['debug_mode'] else 4)

    def test(self, best=True, if_log=True):
        import logging
        self.setup_model()
        self.setup_loss()
        ckpoint_file = os.path.join(self.ckpoint_dir, 'model_best.pth.tar' if best else 'checkpoint.pth.tar')
        last_epoch, best_score = self.initialize_model(self.model, optimizer=None,
                                                       ckpoint_path=ckpoint_file)

        self.setup_test_data()
        dice_per_class, dice_avg = self.eval(self.testing_data_loader)
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
                logging.info("Dice_{}:{:.3f}".format(self.config["class_name"][c], dice_per_class[c]))
            logging.info('\n' + "-" * 50 + '\n')
