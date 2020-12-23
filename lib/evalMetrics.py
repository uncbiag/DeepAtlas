"""
3d segmentation metrics from 
"""
from __future__ import print_function
import numpy as np
import sys
import scipy.spatial
import torch

from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support as score

from .transforms import mask_to_one_hot


#evaluation functions
def metricEval(eval_metric, output, gt, num_labels):
    if eval_metric == 'iou':
        return get_iou(output.squeeze(), gt.squeeze(), num_labels)
    elif eval_metric == 'dice':
        return get_dice(output.squeeze(), gt.squeeze(), num_labels)
    elif eval_metric == 'recall':
        return get_recall(output.squeeze(), gt.squeeze(), num_labels)
    elif eval_metric == 'precision':
        return get_precision(output.squeeze(), gt.squeeze(), num_labels)
    else:
        print('Invalid evaluation metric value')
        sys.exit()
    print('MY IOU', get_iou(output.squeeze(), gt.squeeze(), num_labels))
    print('MY DICE', get_dice(output.squeeze(), gt.squeeze(), num_labels))
    print('MY recall', get_recall(output.squeeze(), gt.squeeze(), num_labels))
    print('MY PRECISION' , get_precision(output.squeeze(), gt.squeeze(), num_labels))
    print(precision_recall_fscore_support(gt.reshape(-1), output.reshape(-1)))

def get_iou(pred, gt, num_labels):
    if pred.shape != gt.shape:
        print('pred shape',pred.shape, 'gt shape', gt.shape)
    assert(pred.shape == gt.shape)
    gt = gt.astype(np.float32)
    pred = pred.astype(np.float32)

    gt = gt.reshape(-1)
    pred = pred.reshape(-1)

    max_label = num_labels-1
    count = np.zeros((max_label+1,))
    for j in range(max_label+1):
        gt_loc = set(np.where(gt == j)[0])
        pred_loc = set(np.where(pred == j)[0])

        intersection = set.intersection(gt_loc, pred_loc)
        union = set.union(gt_loc, pred_loc)

        if len(gt_loc) != 0:
            count[j] = float(len(intersection)) / float(len(union))
    return np.sum(count) / float(num_labels)

def get_dice(pred, gt, num_labels):
    if num_labels != 2:
        print('Dice evaluation score is only implemented for 2 labels')
        sys.exit()
    try:
        dice = 1.0 - scipy.spatial.distance.dice(pred.reshape(-1), gt.reshape(-1))
    except:
        print("Unexpected error:", sys.exc_info()[0])
        dice = 0
        pass
    return dice

#f1 score at beta = 1 is the same as dice score

# recall = (num detected WMH) / (num true WMH)
def get_recall(pred, gt, num_labels):
    if num_labels != 2:
        print('Recall evaluation score is only implemented for 2 labels')
        sys.exit()

    gt = gt.reshape(-1)
    pred = pred.reshape(-1)

    gt_loc = set(np.where(gt == 1)[0])
    pred_loc = set(np.where(pred == 1)[0])
    TP = float(len(set.intersection(gt_loc, pred_loc)))
    TPandFN = float(len(gt_loc))
    return TP / TPandFN

# precision = (number detected WMH) / (number of all detections)
def get_precision(pred, gt, num_labels):
    if num_labels != 2:
        print('Precision evaluation score is only implemented for 2 labels')
        sys.exit()

    gt = gt.reshape(-1)
    pred = pred.reshape(-1)

    gt_loc = set(np.where(gt == 1)[0])
    pred_loc = set(np.where(pred == 1)[0])
    TP = float(len(set.intersection(gt_loc, pred_loc)))
    TPandFP = float(len(pred_loc))
    return TP / TPandFP


def get_multi_metric(pred, gt, eval_label_list=None, rm_bg=False):
    """
    implemented iou, dice, recall, precision metrics for each label of each instance in batch

    :param pred:  predicted(warpped) label map Bx....
    :param gt: ground truth label map  Bx....
    :param eval_label_list: manual selected label need to be evaluate
    :param rm_bg: remove the background label, assume the background label is the first label of label_list when using auto detection
    :return: dictonary, has four items:  multi_metric_res, label_avg_res, batch_avg_res, label_list
    multi_metric_res:{iou: Bx #label , dice: Bx#label...} ,
    label_avg_res:{iou: Bx1 , dice: Bx1...} ,
    batch_avg_res{iou: 1x#label , dice: 1x#label...} ,
    label_list: the labels contained by batch
    """
    # pred = pred.cpu().data.numpy()
    # gt = gt.cpu().data.numpy()
    label_list = np.unique(gt).tolist()
    if rm_bg:
        label_list = label_list[1:]
    if eval_label_list is not None:
        for label in eval_label_list:
            assert label in label_list, "label {} is not in label_list".format(label)
        label_list = eval_label_list
    num_label = len(label_list)
    num_batch = pred.shape[0]
    metrics = ['iou', 'dice', 'recall', 'precision']
    multi_metric_res = {metric: np.zeros([num_batch, num_label]) for metric in metrics}
    label_avg_res = {metric: np.zeros([num_batch, 1]) for metric in metrics}
    batch_avg_res = {metric: np.zeros([1, num_label]) for metric in metrics}

    for l in range(num_label):
        label_pred = (pred == label_list[l]).astype(np.int32)
        label_gt = (gt == label_list[l]).astype(np.int32)
        for b in range(num_batch):
            metric_res = cal_metric(label_pred[b].reshape(-1), label_gt[b].reshape(-1))
            for metric in metrics:
                multi_metric_res[metric][b][l] = metric_res[metric]

    for metric in multi_metric_res:
        for s in range(num_batch):
            no_n_index = np.where(multi_metric_res[metric][s] != -1)
            label_avg_res[metric][s] = float(np.mean(multi_metric_res[metric][s][no_n_index]))

        for l in range(num_label):
            no_n_index = np.where(multi_metric_res[metric][:, l] != -1)
            batch_avg_res[metric][:, l] = float(np.mean(multi_metric_res[metric][:, l][no_n_index]))

    return {'multi_metric_res': multi_metric_res, 'label_avg_res': label_avg_res, 'batch_avg_res': batch_avg_res,
            'label_list': label_list}


def cal_metric(label_pred, label_gt):
    eps = 1e-11
    iou = -1
    recall = -1
    precision = -1
    dice = -1
    gt_loc = set(np.where(label_gt == 1)[0])
    pred_loc = set(np.where(label_pred == 1)[0])
    total_len = len(label_gt)
    # iou
    intersection = set.intersection(gt_loc, pred_loc)
    union = set.union(gt_loc, pred_loc)
    # recall
    len_intersection = len(intersection)
    tp = float(len_intersection)
    tn = float(total_len - len(union))
    fn = float(len(gt_loc) - len_intersection)
    fp = float(len(pred_loc) - len_intersection)

    if len(gt_loc) != 0:
        iou = tp / (float(len(union)) + eps)
        recall = tp / (tp + fn + eps)
        precision = tp / (tp + fp + eps)
        dice = 2 * tp / (2 * tp + fn + fp + eps)

    res = {'iou': iou, 'dice': dice, 'recall': recall, 'precision': precision}

    return res


def get_multiclass_dice(pred, truth, n_class=None, eps=1e-11):
    """
    MultiClass dice score from Pytorch tensor
    :param pred: Tensor of size BxDxMxN, One-hot encoding mask /class-wise probability of prediction
    :param truth: Tensor, ground truth mask when of size BxDxMxN or one-hot mask BxCxDxMxN;
    :return:
    """

    assert pred.shape[0] == truth.shape[0]
    assert pred.shape[-3:] == truth.shape[-3:]



    if n_class is None:
        n_class = max(torch.unique(truth).max(), torch.unique(pred).max()).long().item() + 1

    mask_shape = list(truth.shape)
    pred_one_hot = mask_to_one_hot(pred.view(mask_shape[0], 1, -1), n_class)[:, 1:, :]

    #  if truth are not in onehot size
    if not len(pred.shape) + 1 == len(truth.shape):
        truth_one_hot = mask_to_one_hot(truth.view(mask_shape[0], 1, -1), n_class)[:, 1:, :].float().to(pred_one_hot.device)
    else:
        truth_one_hot = truth.view(mask_shape[0], n_class, -1)[:, 1:, :].float().to(pred_one_hot.device)

    pred_volume = pred_one_hot.sum(2)
    truth_volume = truth_one_hot.sum(2)


    intersection = pred_one_hot * truth_one_hot
    scores = (2. * intersection.sum(2).float() ) / (
            (pred_volume.float() + truth_volume.float()) + eps)

    return scores
