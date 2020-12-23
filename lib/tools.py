import numpy as np
import os
import sys
import shutil
import time
import datetime
import gc
import subprocess
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from torchvision import transforms
# import torchvision.vision_utils as vision_utils
from torch.autograd import Variable
import torch.nn.functional as F

from tensorboardX import SummaryWriter
import SimpleITK as sitk

sys.path.append(os.path.realpath(".."))
import utils.loss as bio_loss
from utils.loss import get_loss_function
import utils.evalMetrics as metrics
import utils.visualize as vis
from misc.module_parameters import save_dict_to_json


def get_identity_transform_batch(size, normalize=True):
    """
    generate an identity transform for given image size (NxCxDxHxW)
    :param size: Batch, D,H,W size
    :param normalize: normalized index into [-1,1]
    :return: identity transform with size Nx3xDxHxW
    """
    _identity = get_identity_transform(size[2:], normalize)
    # return _identity.repeat(size[0], 1, 1, 1, 1)
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

# def weights_init(m):
#     class_name = m.__class__.__name__
#     if class_name.find('Conv') != -1:
#         if not m.weight is None:
#             nn.init.xavier_normal(m.weight.data)
#         if not m.bias is None:
#             nn.init.xavier_normal(m.bias.data)
class write_and_print():
    def __init__(self, if_write, save_dir, log_name):
        self.if_write = if_write
        if if_write:
            self.log = open(os.path.join(save_dir, log_name), 'a')

    def write(self, text):
        if self.if_write:
            self.log.write(text + '\n')
        print(text)

    def close(self):
        if self.if_write:
            self.log.close()


def get_params_num(model):
    """
    Get number of parameters of a reg_model
    :param model: a pytorch modual
    :return: num:int, total number of parameters
    """
    num = 0
    for p in model.parameters():
        num += p.numel()
    return num


def one_hot_encoder(input, num_classes):
    """
    Convert a label tensor/variable into one-hot format
    :param input: A pytorch tensor of size [batchSize,]
    :param num_classes: number of label classes
    :return: output: A pytorch tensor of shape [batchSize x num_classes]
    """
    variable = False
    if isinstance(input, torch.autograd.Variable):
        input = input.data
        variable = True
    output = torch.zeros(input.shape + (num_classes,))
    output.scatter_(1, input.unsqueeze(1), 1.0)
    if variable:
        output = torch.autograd.Variable(output)
    return output


def save_checkpoint(state, is_best, path, prefix=None, name='checkpoint.pth.tar', max_keep=1):
    if not os.path.exists(path):
        os.makedirs(path)
    # name = '_'.join([str(state['epoch']), filename])
    name = '_'.join([prefix, name]) if prefix else name
    best_name = '_'.join([prefix, 'model_best.pth.tar']) if prefix else 'model_best.pth.tar'
    torch.save(state, os.path.join(path, name))
    if is_best:
        torch.save(state, os.path.join(path, best_name))


def weight_from_truth(truths, n_classes):
    ratio_inv = torch.zeros(n_classes)
    for i_class in range(n_classes):
        try:
            ratio_inv[i_class] = len(truths.view(-1)) / torch.sum(truths == i_class)
        except:
            print("Unexpected error:", sys.exc_info()[0])
            ratio_inv[i_class] = 0
            pass
    loss_weight = ratio_inv / torch.sum(ratio_inv)

    return loss_weight






def get_gpu_memory_map():
    """Get the current gpu usage.

    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    # Convert lines into a dictionary
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    return gpu_memory_map

def hookfunc(module, gradInput, gradOutput):
    grd = gradOutput[0]
    for i in range(grd.shape[1]):
        temp_grd = grd[:,i,:,:,:].abs()
        print("class {} min {} max {} median {}".format(i, temp_grd.min(), temp_grd.max(), temp_grd.median()))

# from utils.visualize import plot_grad_flow
def train_model(model, training_data_loader, validation_data_loader, optimizer, args, config, scheduler=None):
    # initialize_model(model, config['resume'])

    # log writer
    now = datetime.datetime.now()
    now_date = "{:02d}{:02d}{:02d}".format(now.month, now.day, now.year)
    now_time = "{:02d}{:02d}{:02d}".format(now.hour, now.minute, now.second)
    ckpoint_name = os.path.join(config['experiment_name'], now_date + '_' + now_time) if not config[
        'debug_mode'] \
        else "debug_seg"

    if not os.path.isdir(os.path.join(config['ckpoint_dir'], ckpoint_name)):
        os.makedirs(os.path.join(config['ckpoint_dir'], ckpoint_name))
    save_dict_to_json(config, os.path.join(config['ckpoint_dir'], ckpoint_name, "train_config.json"))

    total_validation_time = 0
    if config['debug_mode']:
        writer = SummaryWriter(os.path.join(config['log_dir'], 'debug_seg', "{}_{}".format(now_date, now_time)))
    else:
        writer = SummaryWriter(os.path.join(config['log_dir'], config['experiment_name'], now_date + '_' + now_time))

    # if config['loss'] == 'focal_loss':
    #     FC = bio_loss.FocalLoss(config['n_classes'])
    # elif config['loss'] == 'dice':
    #     diceF = bio_loss.DiceLossMultiClass(config['n_classes'], weight_type=config['loss_weight'])
    # elif config['loss'] == 'genDice':
    #     # genDice = bio_loss.GeneralizedDiceLoss4Organs(range(config['n_classes']))
    #     loss_f = bio_loss.GeneralizedDiceLoss(config['n_classes'], weight_type=config['loss_weight'])
    # elif config['loss'] == 'logGenDice':
    #     loss_f = bio_loss.LogGeneralizedDiceLoss(config['n_classes'], weight_type=config['loss_weight'])


    criterion = get_loss_function(config['loss'])(**config['loss_settings']).cuda()
    # model.register_backward_hook(hookfunc)
    # resume checkpoint or initialize
    finished_epochs, best_score = initialize_model(model, optimizer, config['resume_dir'])
    current_epoch = finished_epochs + 1

    # iters_per_epoch = len(training_data_loader.dataset)
    print(config['samples_per_epoch'], config['batch_size'])
    iters_per_epoch = config['samples_per_epoch'] // config['batch_size']

    print("Start Training:")
    while current_epoch <= config['n_epochs']:
        running_loss = 0.0
        is_best = False
        start_time = time.time()  # log running time

        if not config['lr_mode'] == 'const' and not config['lr_mode'] == 'plateau':
            scheduler.step(epoch=current_epoch)

        for i in range(iters_per_epoch):
            if i % len(training_data_loader) == 0:
                print("new with seg iter")
                train_data_iter = iter(training_data_loader)

            model.train()
            optimizer.zero_grad()

            images, truths, name = next(train_data_iter)
            global_step = (current_epoch - 1) * iters_per_epoch + (i + 1) * config['batch_size']  # current globel step

            # weight for loss （inverse ratio of all labels）
            if config['if_weight_loss']:
                if config['weight_mode'] == 'dynamic':
                    loss_weight = weight_from_truth(truths, config['n_classes'])
                elif config['weight_mode'] == 'const':
                    loss_weight = config['weight_const']

            # print(name, truths.min(), truths.max())
            # wrap inputs in Variable
            # images = Variable(images)
            # truths = Variable(truths.long())

            # zero the parameter gradients

            # forward + backward + optimize
            output = model(images.cuda())
            # start_time = time.time()
            # print(time.time() - start_time)

            # output_flat = output.permute(0, 2, 3, 4, 1).contiguous().view(-1, config['n_classes'])
            # truths_flat = truths.view(truths.numel()).long()

            loss = criterion(output, truths.long().cuda())

            # if config['loss'] == 'dice':
            #     loss = diceF(F.softmax(output, 1), truths.long().cuda())
            # elif config['loss'] == 'cross_entropy':
            #     loss = F.cross_entropy(output_flat, truths_flat.cuda(),
            #                            weight=loss_weight.cuda() if config['if_weight_loss'] else None)
            # elif config['loss'] == 'focal_loss':
            #     loss = FC(output_flat, truths_flat.cuda())
            #     # loss = FC(output, truths.long().cuda())
            # elif config['loss'] == 'genDice' or config['loss'] == 'logGenDice':
            #     loss = loss_f(F.softmax(output, 1), truths.long().cuda())
            # # del output_flat, truths_flat
            # else:
            #     ValueError("Undefined loss type: {}".format(config['loss']))

            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()  # average loss over 10 batches
            if i % config['print_batch_period'] == config['print_batch_period'] - 1:  # print every 10 mini-batches
                # print gradient summary
                # for n, p in model.named_parameters():
                #     print(n, p.grad.abs().min().item(), p.grad.abs().median().item(), p.grad.abs().max().item())

                duration = time.time() - start_time
                print('Epoch: {:0d} [{}/{} ({:.0f}%)] loss: {:.3f} lr:{} ({:.3f} sec/batch) {}'.format
                      (current_epoch, (i + 1) * config['batch_size'], iters_per_epoch,
                       (i + 1) * config['batch_size'] / iters_per_epoch * 100,
                       running_loss / config['print_batch_period'] if i > 0 else running_loss,
                       optimizer.param_groups[0]['lr'],
                       duration / config['print_batch_period'],
                       datetime.datetime.now().strftime("%D %H:%M:%S")
                       ))
                writer.add_scalar('loss/training', running_loss / config['print_batch_period'],
                                  global_step=global_step)  # data grouping by `slash`
                writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'],
                                  global_step=global_step)  # data grouping by `slash`
                running_loss = 0.0
                start_time = time.time()  # log running time

        if current_epoch % config['save_ckpts_epoch_period'] == 0:
            image_summary = vis.make_segmentation_image_summary(images, truths, output.cpu())
            writer.add_image("training", image_summary, global_step=global_step)

        # validation
        if current_epoch % config['valid_epoch_period'] == 0:
            with torch.no_grad():
                model.eval()
                dice_per_class = torch.tensor([0.] * (config["n_classes"] - 1))
                start_time = time.time()  # log running time
                valid_data_iter = iter(validation_data_loader)
                for j in range(len(validation_data_loader) if best_score > 0.6 else 5):
                    # for j in range(len(validation_data)):
                    images, truths, name = next(valid_data_iter)

                    if config['if_weight_loss']:
                        if config['weight_mode'] == 'dynamic':
                            loss_weight = weight_from_truth(truths, config['n_classes'])
                        elif config['weight_mode'] == 'const':
                            loss_weight = config['weight_const']

                    valid_batch_size = config['valid_batch_size']

                    # predict through reg_model 1
                    pred = model(images.cuda()).cpu().data

                    for c in range(1, config["n_classes"]):
                        dice_per_class[c - 1] += metrics.metricEval('dice',
                                                                    torch.max(pred, 1)[1].squeeze().numpy() == c,
                                                                    truths.numpy() == c,
                                                                    num_labels=2)

                dice_per_class = dice_per_class / (j + 1)
                dice_avg = dice_per_class.mean()

                if config['lr_mode'] == 'plateau':
                    scheduler.step(dice_avg, epoch=current_epoch)

                is_best = False
                if dice_avg > best_score:
                    is_best = True
                    best_score = dice_avg

                writer.add_scalar('validation_{}/dice_avg'.format(args.data), dice_avg, global_step=global_step)
                for c in range(config["n_classes"] - 1):
                    writer.add_scalar('validation_{}/dice_{}'.format(args.data, config["class_name"][c + 1]), dice_per_class[c],
                                      global_step=global_step)

                image_summary = vis.make_segmentation_image_summary(images, truths, pred)
                writer.add_image("validation", image_summary, global_step=global_step)

                print("Epoch: {:0d} Validation: Dice Avg: {:.4f} ".format(current_epoch, dice_avg) +
                      ' '.join(["Dice_{}:{:.3f}".format(config["class_name"][c + 1], dice_per_class[c]) for c in
                                range(config["n_classes"] - 1)]) +
                      " {:.3f} sec) {}".format(time.time() - start_time,
                                               datetime.datetime.now().strftime("%D %H:%M:%S")))
                total_validation_time += time.time() - start_time

            if current_epoch % config['save_ckpts_epoch_period'] == 0:
                save_checkpoint({'epoch': current_epoch,
                                 'model_state_dict': model.state_dict(),
                                 'optimizer_state_dict': optimizer.state_dict(),
                                 'best_score': best_score},
                                is_best, os.path.join(config['ckpoint_dir'], ckpoint_name))

        current_epoch += 1

    writer.close()
    print('Finished Training: {}_{}_{}'.format(config['experiment_name'], now_date, now_time))
    print('Total validation time: {}'.format(datetime.timedelta(seconds=total_validation_time)))


# eval input using reg_model by iteratively evaluating subsets of a large patch
def pred_iter(model, input, sub_size=4):
    output = []

    for i in range(0, np.ceil(input.size()[0] / sub_size).astype(int)):
        temp_input = input.narrow(0, sub_size * i,
                                  sub_size if sub_size * (i + 1) <= input.size()[0]
                                  else input.size()[0] - sub_size * i)

        temp_input = Variable(temp_input, volatile=True)

        temp_output = model(temp_input.cuda()).data
        output.append(temp_output.cpu())
        del temp_output

    return torch.cat(output, dim=0)
