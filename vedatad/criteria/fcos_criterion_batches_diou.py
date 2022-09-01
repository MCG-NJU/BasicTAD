from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from math import gamma

import sys

import torch
from torch.autograd import Function
from torch.autograd.function import once_differentiable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.distributed as dist
from collections import OrderedDict

def affine_transform(pt, t):     
    new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32).T     
    new_pt = np.dot(t, new_pt)     
    return new_pt[:2]

def batch_bbox_overlaps(bboxes, gts):
    """Calculate overlap between two sets of bboxes

    Args:
        bboxes (Tensor): shape (batchsize, m, 2) in <start, end> format or empty.
        gts (Tensor): shape (batchsize, n, 3) in <start, end> format or empty.

    Returns:
        ious(Tensor): shape (batchsize, m, n)
    """
    batchsize = bboxes.shape[0] 
    M = bboxes.shape[1] 
    N = gts.shape[1] 

    gt_twins = gts[:, :, :2].contiguous() 

    gt_twins_x = (gt_twins[:, :, 1] - gt_twins[:, :, 0] + 1) 
    gt_twins_len = gt_twins_x.view(batchsize, 1, N) 

    bboxes_x = (bboxes[:, :, 1] - bboxes[:, :, 0] + 1)  
    bboxes_len = bboxes_x.view(batchsize, M, 1) 

    gt_len_zero = (gt_twins_x == 1)
    bboxes_len_zero = (bboxes_x == 1)

    twins = bboxes.view(batchsize, M, 1, 2).expand(batchsize, M, N, 2) 
    query_twins = gt_twins.view(batchsize, 1, N, 2).expand(batchsize, M, N, 2) 

    ilen = (torch.min(twins[:, :, :, 1], query_twins[:, :, :, 1]) -
            torch.max(twins[:, :, :, 0], query_twins[:, :, :, 0]) + 1)
    ilen[ilen < 0] = 0  # batchsize*M*N

    ua = bboxes_len + gt_twins_len - ilen
    overlaps = ilen / ua

    # mask the overlap
    overlaps.masked_fill_(gt_len_zero.view(batchsize, 1, N).expand(batchsize, M, N), -1)
    overlaps.masked_fill_(bboxes_len_zero.view(batchsize, M, 1).expand(batchsize, M, N), -1)

    return overlaps

def offsets2bboxes(offsets, down_ratio):
    """ map offset to bbox in the original video

    Args:
        offsets (Tensor): shape (N, T, 2)

    return:
        bboxes (Tensor): shape (N, T, 2)
    """
    output_length = offsets.shape[1] 
    tmp_locations = torch.arange(0, output_length, dtype=offsets.dtype,
                                 device=offsets.device) + 0.5
    tmp_locations = tmp_locations.repeat(offsets.shape[0], 1)  # N*T
    det_bboxes = torch.zeros_like(offsets)  # N*T*2
    det_bboxes[:, :, 0] = tmp_locations - offsets[:, :, 0]
    det_bboxes[:, :, 1] = tmp_locations + offsets[:, :, 1]

    det_bboxes = det_bboxes * down_ratio
    det_bboxes = det_bboxes.clamp_(min=0.0, max=output_length*down_ratio)  # N*T*2

    return det_bboxes

def batch_bbox_transform(roi_bboxes, gt_bboxes):
    """Calculate bounding box regression targets for each roi box

    Args:
        bboxes (Tensor): shape (batchsize, m, 2) in <start, end>
        gt_bboxes (Tensor): shape (batchsize, m, 2) in <start, end>

    Returns:
        targets (Tensor): shape (batchsize, m, 2)
    """

    roi_lens = roi_bboxes[:, :, 1] - roi_bboxes[:, :, 0] + 1.0
    roi_ctr_x = roi_bboxes[:, :, 0] + 0.5 * roi_lens

    gt_lens = gt_bboxes[:, :, 1] - gt_bboxes[:, :, 0] + 1.0
    gt_ctr_x = gt_bboxes[:, :, 0] + 0.5 * gt_lens

    targets_dx = (gt_ctr_x - roi_ctr_x) / roi_lens
    targets_dl = torch.log(gt_lens / roi_lens)

    targets = torch.stack((targets_dx, targets_dl), 2)

    return targets

def _smooth_l1_loss(bbox_pred, bbox_targets, sigma=1.0, dim=1):

    if torch.numel(bbox_pred)==0:
        return 0

    sigma_2 = sigma ** 2
    box_diff = bbox_pred - bbox_targets  
    abs_box_diff = torch.abs(box_diff)
    smoothL1_sign = (abs_box_diff < 1. / sigma_2).detach().float()
    loss_box = torch.pow(box_diff, 2) * (sigma_2 / 2.) * smoothL1_sign \
               + (abs_box_diff - (0.5 / sigma_2)) * (1. - smoothL1_sign)
    loss_box = loss_box.sum(dim)
    loss_box = loss_box.mean()
    return loss_box

def _gather_feat(feat, ind, mask=None):
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat
def _transpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat



def one_hot(labels, num_class, eps=0.0):
    if not torch.is_tensor(labels):
        raise TypeError('Input labels type is not a torch.Tensor. Got {}'.format(type(labels)))

    if not len(labels.shape) == 3:
        raise TypeError('Invalid depth shape, we expect BxHxW. Got {}'.format(labels.shape))

    if num_class < 1:
        raise ValueError('The number of classes must be bigger than one. Got {}'.format(num_class))

    batch_size, height, width = labels.shape
    one_hot = torch.zeros(batch_size, num_class, height, width, device=labels.device)

    return one_hot.scatter_(1, labels.unsqueeze(1).long(), 1.0) + eps


def focal_loss(input, target, alpha=0.75, gamma=2.0, reduction='none', eps=1e-8):

    if not torch.is_tensor(input):
        raise TypeError('Input type is not a torch.Tensor. Got {}'.format(type(input)))

    if not len(input.shape) == 4:
        raise ValueError('Invalid input shape, we expect BxNxHxW. Got {}'.format(input.shape))

    if not input.shape[-2:] == target.shape[-2:]:
        raise ValueError(
            'input and target shapes must be the same. Got: {} {}'.format(
                input.shape, input.shape
            )
        )

    input_soft = F.softmax(input, dim=1) + eps
    target_one_hot = one_hot(target, num_class=input.shape[1])
    weight = torch.pow(-input_soft + 1., gamma)

    focal = -alpha * weight * torch.log(input_soft)
    loss_tmp = torch.sum(target_one_hot * focal, dim=1)

    if reduction == 'none':
        loss = loss_tmp
    elif reduction == 'mean':
        loss = torch.mean(loss_tmp)
    elif reduction == 'sum':
        loss = torch.sum(loss_tmp)
    else:
        raise NotImplementedError('Invalid reduction mode: {}'.format(reduction))

    return loss


class FocalLoss(nn.Module):

    def __init__(self, alpha=0.75, gamma=2.0, reduction='none'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.eps = 1e-6

    def forward(self, input, target):

        pos=torch.nonzero(target!=-1)
        target=target[pos].squeeze()
        input=input[pos,:].squeeze()
        
        if len(input.shape) == 2:
            input = input.reshape(input.shape[0], input.shape[1], 1, 1)
            target = target.reshape(target.shape[0], 1, 1)

        return focal_loss(
            input,
            target,
            alpha=self.alpha,
            gamma=self.gamma,
            reduction=self.reduction,
            eps=self.eps
        )


def sigmoid_focal_loss_cpu(pred, gt, gamma, alpha):
    """binary focal loss from FCOS

    Arguments:
        pred: (N*T)*C
        gt: (N*T),
    """
    num_classes = pred.shape[1]
    dtype = gt.dtype
    device = gt.device
    class_range = torch.arange(1, num_classes + 1, dtype=dtype, device=device).unsqueeze(0)

    t = gt.unsqueeze(1)
    p = torch.sigmoid(pred)

    term1 = (1 - p)**gamma * torch.log(p + 1e-7)
    term2 = p**gamma * torch.log(1 - p + 1e-7)
    loss = -(t == class_range).float() * term1 * alpha \
           - ((t != class_range) * (t >= 0)).float() * term2 * (1 - alpha)

    return loss

def _reg_loss(regr, gt_regr, mask):
    """ L1 regression loss
    
    Arguments:
        regr (batch x max_objects x dim)
        gt_regr (batch x max_objects x dim)
        mask (batch x max_objects)
    """
    num = mask.float().sum()
    mask = mask.unsqueeze(2).expand_as(gt_regr).float()
    
    regr = regr * mask
    gt_regr = gt_regr * mask
    
    regr_loss = nn.functional.smooth_l1_loss(regr, gt_regr, size_average=False)
    regr_loss = regr_loss / (num + 1e-4)
    return regr_loss

class IOULoss(nn.Module):
    """ IoU, GIoU, and Linear IoU loss from FOCS

    Arguments:
        pred: (N*T')*2,
        target: (N*T')*2,

        T':only points inside the gt windows are utilized for calculating IoU loss
    """

    def __init__(self, loss_type='iou', centerness=True):
        super(IOULoss, self).__init__()
        self.loss_type = loss_type
        self.centerness = centerness

    def forward(self, pred, target):
        # import pdb; pdb.set_trace()
        pred_start_offset = pred[:, 0]     # x-start
        pred_end_offset = pred[:, 1]     # end-x

        target_start_offset = target[:, 0]     # x-start
        target_end_offset = target[:, 1]     # end-x

        pred_length = pred_start_offset + pred_end_offset     # (x-start) + (end-x) = end - start
        target_length = target_start_offset + target_end_offset

        length_intersect = (torch.min(pred_start_offset, target_start_offset)
                            + torch.min(pred_end_offset, target_end_offset))

        centerness_weight = (
            torch.sqrt(
                (torch.min(target_start_offset, target_end_offset)
                 / (torch.max(target_start_offset, target_end_offset) + 1e-5))
            )
        )

        length_circum = (
            torch.max(pred_start_offset, target_start_offset) + torch.max(pred_end_offset, target_end_offset) + 1e-5
        )

        length_union = pred_length + target_length - length_intersect

        if self.centerness:
            centerness_weight = (
                torch.sqrt(
                    (torch.min(target_start_offset, target_end_offset)
                     / (torch.max(target_start_offset, target_end_offset) + 1e-5))
                )
            )
            ious = (length_intersect + 1e-5) / (length_union + 1e-5)
            losses = -torch.log(ious)
            assert losses.numel() > 0
            loss = (losses * centerness_weight).sum() / (centerness_weight.sum() + 1e-5)
            return loss
        else:
            ious = (length_intersect + 1.0) / (length_union + 1.0)
            gious = ious - (length_circum - length_union) / length_circum

            if self.loss_type == 'iou':
                losses = -torch.log(ious)
            elif self.loss_type == 'linear_iou':
                losses = 1 - ious
            elif self.loss_type == 'giou':
                losses = 1 - gious
            else:
                raise NotImplementedError
            assert losses.numel() > 0
            loss = losses.sum()
            return loss

class RegLoss(nn.Module):
    """Regression loss for an output tensor
    
    Arguments:
        output (batch x dim x h x w)
        mask (batch x max_objects)
        ind (batch x max_objects)
        target (batch x max_objects x dim)
    """
    
    def __init__(self):
        super(RegLoss, self).__init__()
        
    def forward(self, output, mask, ind, target):
        pred = _transpose_and_gather_feat(output, ind)
        loss = _reg_loss(pred, target, mask)
        return loss

class RegL1Loss(nn.Module):

    def __init__(self):
        super(RegL1Loss, self).__init__()
        
    def forward(self, output, mask, ind, target):
        # import pdb; pdb.set_trace()
        pred = _transpose_and_gather_feat(output, ind)
        mask = mask.unsqueeze(2).expand_as(pred).float()
        loss = F.l1_loss(pred * mask, target * mask, reduction='sum')
        loss = loss / (mask.sum() + 1e-4)
        return loss

class NormRegL1Loss(nn.Module):
    
    def __init__(self):
        super(NormRegL1Loss, self).__init__()
        
    def forward(self, output, mask, ind, target):
        pred = _transpose_and_gather_feat(output, ind)
        mask = mask.unsqueeze(2).expand_as(pred).float()
        pred = pred / (target + 1e-4)
        target = target * 0 + 1
        loss = F.l1_loss(pred * mask, target * mask, size_average=False)
        loss = loss / (mask.sum() + 1e-4)
        return loss

class RegWeightedL1Loss(nn.Module):
    
    def __init__(self):
        super(RegWeightedL1Loss, self).__init__()
        
    def forward(self, output, mask, ind, target):
        pred = _transpose_and_gather_feat(output, ind)
        mask = mask.float()
        loss = F.l1_loss(pred * mask, target * mask, size_average=False)
        loss = loss / (mask.sum() + 1e-4)
        return loss

class L1Loss(nn.Module):
    
    def __init__(self):
        super(L1Loss, self).__init__()
    
    def forward(self, output, mask, ind, target):
        pred = _transpose_and_gather_feat(output, ind)
        mask = mask.unsqueeze(2).expand_as(pred).float()
        loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
        return loss

class BinRotLoss(nn.Module):
    
    def __init__(self):
        super(BinRotLoss, self).__init__()
        
    def forward(self, output, mask, ind, rotbin, rotres):
        pred = _transpose_and_gather_feat(output, ind)
        loss = compute_rot_loss(pred, rotbin, rotres, mask)
        return loss


def compute_res_loss(output, target):
    return F.smooth_l1_loss(output, target, reduction='elementwise_mean')

# TODO: weight
def compute_bin_loss(output, target, mask):
    mask = mask.expand_as(output)
    output = output * mask.float()
    return F.cross_entropy(output, target, reduction='elementwise_mean')

def compute_rot_loss(output, target_bin, target_res, mask):
    # output: (B, 128, 8) [bin1_cls[0], bin1_cls[1], bin1_sin, bin1_cos, 
    #                 bin2_cls[0], bin2_cls[1], bin2_sin, bin2_cos]
    # target_bin: (B, 128, 2) [bin1_cls, bin2_cls]
    # target_res: (B, 128, 2) [bin1_res, bin2_res]
    # mask: (B, 128, 1)
    # import pdb; pdb.set_trace()
    output = output.view(-1, 8)
    target_bin = target_bin.view(-1, 2)
    target_res = target_res.view(-1, 2)
    mask = mask.view(-1, 1)
    loss_bin1 = compute_bin_loss(output[:, 0:2], target_bin[:, 0], mask)
    loss_bin2 = compute_bin_loss(output[:, 4:6], target_bin[:, 1], mask)
    loss_res = torch.zeros_like(loss_bin1)
    if target_bin[:, 0].nonzero().shape[0] > 0:
        idx1 = target_bin[:, 0].nonzero()[:, 0]
        valid_output1 = torch.index_select(output, 0, idx1.long())
        valid_target_res1 = torch.index_select(target_res, 0, idx1.long())
        loss_sin1 = compute_res_loss(
          valid_output1[:, 2], torch.sin(valid_target_res1[:, 0]))
        loss_cos1 = compute_res_loss(
          valid_output1[:, 3], torch.cos(valid_target_res1[:, 0]))
        loss_res += loss_sin1 + loss_cos1
    if target_bin[:, 1].nonzero().shape[0] > 0:
        idx2 = target_bin[:, 1].nonzero()[:, 0]
        valid_output2 = torch.index_select(output, 0, idx2.long())
        valid_target_res2 = torch.index_select(target_res, 0, idx2.long())
        loss_sin2 = compute_res_loss(
          valid_output2[:, 6], torch.sin(valid_target_res2[:, 1]))
        loss_cos2 = compute_res_loss(
          valid_output2[:, 7], torch.cos(valid_target_res2[:, 1]))
        loss_res += loss_sin2 + loss_cos2
    return loss_bin1 + loss_bin2 + loss_res

class Diou_loss(nn.Module):
  def __init__(self,eps=1e-6,centerness=True):
        super(Diou_loss, self).__init__()
        self.centerness = centerness
        self.eps=eps
  def forward(self,pred, target):
    """`Implementation of Distance-IoU Loss: Faster and Better
    Learning for Bounding Box Regression, https://arxiv.org/abs/1911.08287`_.
    Code is modified from https://github.com/Zzh-tju/DIoU.
    Args:
        pred (Tensor): Predicted segments of format (start, end),
            shape (n, 2).
        target (Tensor): Corresponding gt segments, shape (n, 2).
        eps (float): Eps to avoid log(0).
    Return:
        Tensor: Loss tensor.
    """
    pred_start_offset = pred[:, 0]     # x-start  
    pred_end_offset = pred[:, 1]     # end-x

    target_start_offset = target[:, 0]     # x-start
    target_end_offset = target[:, 1]     # end-x

    pred_length = pred_start_offset + pred_end_offset # (x-start) + (end-x) = end - start
    target_length = target_start_offset + target_end_offset 

    length_intersect = (torch.min(pred_start_offset, target_start_offset)
                            + torch.min(pred_end_offset, target_end_offset)) 
    length_union = pred_length + target_length - length_intersect + self.eps 

    length_circum = (
            torch.max(pred_start_offset, target_start_offset) + torch.max(pred_end_offset, target_end_offset) + self.eps
        )
    c2 = length_circum**2 + self.eps

    ious=length_intersect/length_union

    pred_start=-pred[:,0]
    pred_end=pred[:,1]
    target_start=-target[:,0]
    target_end=target[:,1]
    pred_center = (pred_start + pred_end) * 0.5
    target_center = (target_start + target_end) * 0.5
    rho2 = (target_center - pred_center)**2
    # DIoU
    dious = ious - rho2 / c2
    loss = 1 - dious

    loss_final = loss.mean()

    return loss_final

from mmcv.utils import ext_loader
ext_module = ext_loader.load_ext('_ext', [
    'sigmoid_focal_loss_forward', 'sigmoid_focal_loss_backward',
    'softmax_focal_loss_forward', 'softmax_focal_loss_backward'
])
class _SigmoidFocalLoss(Function):

    @staticmethod
    def symbolic(g, input, target, gamma, alpha, weight, reduction):
        return g.op(
            'mmcv::MMCVSigmoidFocalLoss',
            input,
            target,
            gamma_f=gamma,
            alpha_f=alpha,
            weight_f=weight,
            reduction_s=reduction)

    @staticmethod
    def forward(ctx,
                input,
                target,
                gamma=2.0,
                alpha=0.25,
                weight=None,
                reduction='mean'):

        assert isinstance(target, (torch.LongTensor, torch.cuda.LongTensor))
        assert input.dim() == 2
        assert target.dim() == 1
        assert input.size(0) == target.size(0)
        if weight is None:
            weight = input.new_empty(0)
        else:
            assert weight.dim() == 1
            assert input.size(1) == weight.size(0)
        ctx.reduction_dict = {'none': 0, 'mean': 1, 'sum': 2}
        assert reduction in ctx.reduction_dict.keys()

        ctx.gamma = float(gamma)
        ctx.alpha = float(alpha)
        ctx.reduction = ctx.reduction_dict[reduction]

        output = input.new_zeros(input.size())

        ext_module.sigmoid_focal_loss_forward(
            input, target, weight, output, gamma=ctx.gamma, alpha=ctx.alpha)
        
        if ctx.reduction == ctx.reduction_dict['mean']:
            output = output.sum() / input.size(0)
        elif ctx.reduction == ctx.reduction_dict['sum']:
            output = output.sum()
        ctx.save_for_backward(input, target, weight)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        input, target, weight = ctx.saved_tensors

        grad_input = input.new_zeros(input.size())

        ext_module.sigmoid_focal_loss_backward(
            input,
            target,
            weight,
            grad_input,
            gamma=ctx.gamma,
            alpha=ctx.alpha)

        grad_input *= grad_output
        if ctx.reduction == ctx.reduction_dict['mean']:
            grad_input /= input.size(0)
        return grad_input, None, None, None, None, None

class SigmoidFocalLoss(nn.Module):
    """nn.Module warpper for sigmoid focal loss
    """

    def __init__(self,alpha,gamma):
        super(SigmoidFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, out, target):

        pos=torch.nonzero(target!=-1)
        target=target[pos].squeeze()
        out=out[pos,:].squeeze()

        loss_func = _SigmoidFocalLoss.apply
        loss = loss_func(out, target.long(), self.gamma, self.alpha)
        return loss.sum()

class FocalLoss_single(nn.Module):
    def __init__(self, alpha=0.75, gamma=2, reduction='none'):
        super(FocalLoss_single, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.eps = 1e-6
    
    def forward(self, input, target):
        
        pos=torch.nonzero(target!=-1)
        target=target[pos].squeeze()    
        input=input[pos,:].squeeze()
        
        input_soft = F.softmax(input, dim=1).view(-1,2) 

        target = target.view(-1,1)
        target[target>1]=1
        class_mask = torch.zeros(input_soft.shape[0],input_soft.shape[1]).to(input_soft.device)
        class_mask.scatter_(1, target.view(-1, 1).long(), 1.)
        
        probs = (input_soft * class_mask).sum(dim=1).view(-1,1)
        probs = probs.clamp(min=0.0001,max=1.0)
        
        log_p = probs.log()
        
        alpha = torch.ones(input_soft.shape[0],input_soft.shape[1]).to(input_soft.device)
        alpha[:,0] = alpha[:,0] * (1-self.alpha)
        alpha[:,1] = alpha[:,1] * self.alpha
        alpha = (alpha * class_mask).sum(dim=1).view(-1,1)
        
        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p
        
        if self.reduction == 'none':
            loss = batch_loss
        elif self.reduction == 'mean':
            loss = torch.mean(batch_loss)
        elif self.reduction == 'sum':
            loss = torch.sum(batch_loss)
        else:
            raise NotImplementedError('Invalid reduction mode: {}'.format(self.reduction))

        return loss



from vedacore.misc import multi_apply, registry, unmap
@registry.register_module('criterion')
class FcosActFPNContextRegLossCriterion_batches_diou(torch.nn.Module):
    def __init__(self,act_loss_type,iou_loss_type,act_cls,batch_size,down_ratio_list,num_stacks,reg_range_list,num_max_acts,is_thumos):
        super(FcosActFPNContextRegLossCriterion_batches_diou, self).__init__()
        self.act_loss_type=act_loss_type
        if act_loss_type == 'sigmoidfocal':
            self.crit_act_cls = SigmoidFocalLoss(alpha=0.25,gamma=2.0)
        elif act_loss_type == 'softmax':
            self.crit_act_cls = torch.nn.CrossEntropyLoss(reduction='mean', ignore_index=-1)
        elif act_loss_type == 'focal':
            self.crit_act_cls = FocalLoss(alpha=0.75, gamma=2.0, reduction='mean')
        elif act_loss_type == 'focal_single':             
            self.crit_act_cls = FocalLoss_single(alpha=0.75, gamma=2.0, reduction='mean')
        else:
            raise NotImplementedError

        self.act_cls=act_cls
        self.batch_size=batch_size

        self.crit_offset_reg = Diou_loss(eps=1e-6,centerness=True)
        self.crit_centerness = torch.nn.BCEWithLogitsLoss(reduction='sum')
        self.crit_roi_cls = torch.nn.CrossEntropyLoss(reduction='mean', ignore_index=-1)

        self.down_ratio_list=down_ratio_list
        self.num_stacks=num_stacks
        self.reg_range_list=reg_range_list
        self.num_max_acts=num_max_acts
        self.is_thumos=is_thumos
    
    @staticmethod
    def _parse_losses(losses):
        """Parse the raw outputs (losses) of the network.

        Adapted from https://github.com/open-mmlab/mmdetection

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary infomation.
        Returns:
            tuple[Tensor, dict]: (loss, log_vars), loss is the loss tensor \
                which may be a weighted sum of all losses, log_vars contains \
                all the variables to be sent to the logger.
        """
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        loss = sum(_value for _key, _value in log_vars.items()
                   if 'loss' in _key)

        log_vars['loss'] = loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars


    def loss(self,feats, video_metas, gt_segments,gt_labels, gt_segments_ignore):  

        outputs=feats
        down_ratio_list=self.down_ratio_list
        num_stacks=self.num_stacks

        act_cls_label_list=[]
        offset_reg_target_list=[]
        for ii in range(len(gt_segments)):
            video_length=video_metas[ii]['pad_tsize']

            gt_bboxes=gt_segments[ii].detach().cpu()
            gt_labels_2=gt_labels[ii].detach().cpu()

            gt_lengths=gt_bboxes[:, 1] - gt_bboxes[:, 0]
            gt_classes=gt_labels_2
        
            gt_inds = np.where(gt_lengths > 0)[0]
            gt_bboxes = gt_bboxes[gt_inds, :]
            gt_lengths = gt_lengths[gt_inds]
            gt_classes = gt_classes[gt_inds]

            if gt_bboxes.shape[0] == 0:
                gt_bboxes = np.zeros((1, 2))
                gt_lengths = np.zeros(1)
                gt_classes = np.ones(1)
        
            num_acts = gt_bboxes.shape[0] 

            # deal with diving and clifdiving
            if self.is_thumos:
                delete_ind = []
                for i in range(num_acts-1):
                    gt_bbox = gt_bboxes[i]
                    gt_cls = gt_classes[i]
                    next_bbox = gt_bboxes[i+1]
                    next_gt_cls = gt_classes[i+1]

                    if (gt_bbox == next_bbox).all():
                        if (gt_cls + next_gt_cls) == 13:
                            if gt_cls == 8:
                                delete_ind.append(i)
                            if next_gt_cls == 8:
                                delete_ind.append(i+1)

                gt_bboxes = np.delete(gt_bboxes, delete_ind, axis=0)
                gt_classes = np.delete(gt_classes, delete_ind, axis=0)
                gt_lengths = np.delete(gt_lengths, delete_ind, axis=0)
                num_acts = gt_bboxes.shape[0]
            
            assert (num_acts < self.num_max_acts)


            output_length_list = [video_length // down_ratio for down_ratio in down_ratio_list]
            INF = float("inf")
            reg_range_list = self.reg_range_list
            reg_range_list[-1][1]=INF

            num_lvls = len(output_length_list) 
            total_output_length = 0
            for i in range(num_lvls):
                total_output_length += output_length_list[i]

            act_cls_label = np.zeros((total_output_length,), dtype=np.float32) 
            offset_reg_target = np.zeros((total_output_length, 2), dtype=np.float32) 
        
            multicls_offset_reg_target = np.zeros((total_output_length, num_acts, 2), dtype=np.float32) 
            multicls_reg_range = np.zeros((total_output_length, num_acts, 2), dtype=np.float32)

            tmp_locations_list = []
            trans_output_list = []

            for i in range(num_lvls): 
                output_length = output_length_list[i]
                down_ratio = down_ratio_list[i]

                tmp_locations = np.arange(0, output_length) + 0.5

                trans_output = np.array(
                    [[1. / down_ratio, 0, 0],
                     [0, 1. / down_ratio, 0]])


                tmp_locations_list.append(tmp_locations)
                trans_output_list.append(trans_output)
        
            for k in range(num_acts):
                for i in range(num_lvls):
                    tmp_locations = tmp_locations_list[i]
                    trans_output = trans_output_list[i] 
                    gt_box = gt_bboxes[k] 
                    gt_box = affine_transform(gt_box[0:2], trans_output) 
                    if i == 0:
                        start_point = 0
                    end_point = start_point + output_length_list[i] 

                    start_offset = tmp_locations[:] - gt_box[0]
                    end_offset = gt_box[1] - tmp_locations[:]
                    multicls_offset_reg_target[start_point:end_point, k, 0] = start_offset[:]
                    multicls_offset_reg_target[start_point:end_point, k, 1] = end_offset[:]

                    multicls_reg_range[start_point:end_point, :, 0] = reg_range_list[i][0]
                    multicls_reg_range[start_point:end_point, :, 1] = reg_range_list[i][1]

                    start_point = end_point
        
            is_in_gts = multicls_offset_reg_target.min(axis=2) >= 0 
            tmp_locations_to_gt_lengths = np.tile(gt_lengths, (total_output_length, 1))
            tmp_locations_to_gt_lengths[is_in_gts == 0] = INF
            tmp_locations_to_min_lengths = np.min(tmp_locations_to_gt_lengths, axis=1)
            tmp_locations_to_gt_inds = np.argmin(tmp_locations_to_gt_lengths, axis=1)
            offset_reg_target = multicls_offset_reg_target[range(total_output_length), tmp_locations_to_gt_inds]
            act_cls_label = gt_classes[tmp_locations_to_gt_inds] 
            act_cls_label[tmp_locations_to_min_lengths == INF] = 0 

            max_offset_reg = multicls_offset_reg_target.max(axis=2)
            inside_reg_range = np.logical_and((max_offset_reg >= multicls_reg_range[..., 0]),
                                          (max_offset_reg <= multicls_reg_range[..., 1]),)

            tmp_locations_to_gt_lengths[inside_reg_range == 0] = INF
            # if there are more than one tmp window for a tmp location,
            # we choose the one with minimal area
            tmp_locations_to_min_lengths = np.min(tmp_locations_to_gt_lengths, axis=1)
            tmp_locations_to_gt_inds = np.argmin(tmp_locations_to_gt_lengths, axis=1)
        
            # import pdb; pdb.set_trace()
            offset_reg_target = multicls_offset_reg_target[range(total_output_length), tmp_locations_to_gt_inds]
            fpn_act_cls_label = gt_classes[tmp_locations_to_gt_inds]
            fpn_act_cls_label[tmp_locations_to_min_lengths == INF] = 0
            is_ignore = np.logical_and((act_cls_label > 0), (fpn_act_cls_label == 0))
            act_cls_label[is_ignore] = -1

            if self.is_thumos:
            # deal with the 'CliffDiving' and 'Diving' problem
                pos_inds = np.where(act_cls_label > 0)[0]
                for l in pos_inds:
                    single_tmp_in_gt = is_in_gts[l]
                    single_tmp_gt_classes = gt_classes[single_tmp_in_gt]
                    if 5 in single_tmp_gt_classes and 8 in single_tmp_gt_classes:
                        act_cls_label[l] = 5

            act_cls_label=act_cls_label.to(gt_segments[0].device)
            offset_reg_target=torch.from_numpy(offset_reg_target)
            offset_reg_target=offset_reg_target.to(gt_labels[0].device)

            act_cls_label_list.append(act_cls_label)
            offset_reg_target_list.append(offset_reg_target)
            

        act_cls_loss_list=0
        offset_reg_loss_list=0
        for i in range(len(act_cls_label_list)):
            act_cls_loss, offset_reg_loss = 0, 0

            act_cls_label_flatten = act_cls_label_list[i].reshape(-1)
            offset_reg_target_flatten = offset_reg_target_list[i].reshape(-1,2)

            act_cls_flatten = []
            offset_reg_flatten = []
            coeff_list = []

            start_loc = 0
            for s in range(num_stacks):
                output = outputs[s]
            
                output_act_cls=output['act_cls'][i,:,:].unsqueeze(0)
                output_offset_reg=output['offset_reg'][i,:,:].unsqueeze(0)

                single_act_cls = output_act_cls.permute(0, 2, 1) 
                act_cls_flatten.append(single_act_cls)

                single_offset_reg = output_offset_reg.permute(0, 2, 1)
                offset_reg_flatten.append(single_offset_reg)

            act_cls_flatten = torch.cat(act_cls_flatten, dim=1)  # N*T*C
            act_cls_flatten = act_cls_flatten.reshape(-1, self.act_cls)
            offset_reg_flatten = torch.cat(offset_reg_flatten, dim=1)  # N*T*2
            offset_reg_flatten = offset_reg_flatten.reshape(-1, 2)

            pos_inds = torch.nonzero(act_cls_label_flatten > 0).view(-1)
            num_pos = pos_inds.clone().detach().numel()
            num_pos = float(num_pos)
            neg_inds = torch.nonzero(act_cls_label_flatten == 0).view(-1)
            num_neg = neg_inds.clone().detach().numel()
            num_neg = float(num_neg)

            offset_reg_flatten = offset_reg_flatten[pos_inds]
            offset_reg_target_flatten = offset_reg_target_flatten[pos_inds]

            if self.act_loss_type == 'sigmoidfocal':
                act_cls_label_flatten = act_cls_label_flatten.type(torch.long)
                act_cls_loss += self.crit_act_cls(act_cls_flatten, act_cls_label_flatten) 
            if self.act_loss_type == 'softmax' or self.act_loss_type == 'focal' or self.act_loss_type == 'focal_single':
                act_cls_label_flatten = act_cls_label_flatten.type(torch.long)
                act_cls_loss += self.crit_act_cls(act_cls_flatten, act_cls_label_flatten)
        
            if num_pos > 0:
                if self.crit_offset_reg.centerness:
                    offset_reg_loss += self.crit_offset_reg(offset_reg_flatten, offset_reg_target_flatten)
                else:
                    offset_reg_loss += self.crit_offset_reg(offset_reg_flatten, offset_reg_target_flatten) / num_pos

            else:
                offset_reg_loss += torch.tensor(0)
                offset_reg_loss = offset_reg_loss.type(torch.float)
                offset_reg_loss=offset_reg_loss.to(offset_reg_flatten.device)


            loss = act_cls_loss +  offset_reg_loss 
            act_cls_loss_list+=act_cls_loss
            offset_reg_loss_list+=offset_reg_loss
        
        act_cls_loss_list=act_cls_loss_list/len(act_cls_label_list)
        offset_reg_loss_list=offset_reg_loss_list/len(act_cls_label_list)

        loss, log_vars = self._parse_losses(
            dict(loss_cls=act_cls_loss_list, loss_segment=offset_reg_loss_list))
        return dict(loss=loss, log_vars=log_vars)


