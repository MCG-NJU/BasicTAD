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

INF = 1e8

def affine_transform(pt, t):     
    new_pt = torch.tensor([pt[0], pt[1], 1.]).T.to(pt.device)     
    new_pt = t.mul(new_pt).to(pt.device)      
    return new_pt[:2]

def distance2bbox(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.

    Args:
        points (Tensor): Shape (B, N, 2) or (N, 2).
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom). Shape (B, N, 4) or (N, 4)
        max_shape (Sequence[int] or torch.Tensor or Sequence[
            Sequence[int]],optional): Maximum bounds for boxes, specifies
            (H, W, C) or (H, W). If priors shape is (B, N, 4), then
            the max_shape should be a Sequence[Sequence[int]]
            and the length of max_shape should also be B.

    Returns:
        Tensor: Boxes with shape (N, 4) or (B, N, 4)
    """
    x1 = points[..., 0] - distance[..., 0]
    x2 = points[..., 0] + distance[..., 1]
    bboxes = torch.stack([x1,x2], -1)

    if max_shape is not None:
        if not isinstance(max_shape, torch.Tensor):
            max_shape = x1.new_tensor(max_shape)
        max_shape = max_shape[..., :1].type_as(x1)
        min_xy = x1.new_tensor(0)
        max_xy = torch.cat([max_shape],dim=-1).flip(-1).unsqueeze(-2)
        bboxes = torch.where(bboxes < min_xy, min_xy, bboxes)
        bboxes = torch.where(bboxes > max_xy, max_xy, bboxes)

    return bboxes


from vedacore.misc import multi_apply, registry, unmap,reduce_mean
from .losses import build_loss
@registry.register_module('criterion')
class FcosCriterion(torch.nn.Module):
    def __init__(self,reg_range_list,loss_cls,loss_segment,num_classes,strides,is_thumos):
        super(FcosCriterion, self).__init__()
        self.reg_range_list=reg_range_list
        self.is_thumos=is_thumos
        self.loss_cls = build_loss(loss_cls)
        self.loss_segment = build_loss(loss_segment)
        self.cls_out_channels = num_classes
        self.strides=strides
        self.num_classes=num_classes
    
    def init_weights(self):
        """Initialize weights of the head."""
        pass
    
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
    
    def _get_points_single(self,
                           featmap_size,
                           stride,
                           dtype,
                           device,
                           flatten=False):
        """Get points of a single scale level."""
        w = featmap_size

        x_range = torch.arange(w, device=device).to(dtype)
        
        if flatten:
            x_range = x_range.flatten()
        
        points = x_range + 0.5
        points_concate=torch.cat((torch.unsqueeze(points,-1),torch.unsqueeze(points,-1)),dim=-1)
        return points,points_concate
    
    def get_points(self, featmap_sizes, dtype, device, flatten=False):
        """Get points according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            dtype (torch.dtype): Type of points.
            device (torch.device): Device of points.

        Returns:
            tuple: points of each image.
        """
        mlvl_points = []
        mlvl_points_concate = []
        for i in range(len(featmap_sizes)):
            points,points_concate=self._get_points_single(featmap_sizes[i], self.strides[i],
                                        dtype, device, flatten)
            mlvl_points.append(points)
            mlvl_points_concate.append(points_concate)

        return mlvl_points,mlvl_points_concate
    
    def _get_target_single(self, gt_bboxes, gt_labels, points, regress_ranges,
                           num_points_per_lvl):
        num_points = points.size(0)
        num_gts = gt_labels.size(0)

        if num_gts == 0:
            return gt_labels.new_full((num_points,), self.num_classes), \
                   gt_bboxes.new_zeros((num_points, 2))
        areas = (gt_bboxes[:, 1] - gt_bboxes[:, 0])
        areas = areas[None].repeat(num_points, 1)
        regress_ranges = regress_ranges[:, None, :].expand(
            num_points, num_gts, 2)

        for k in range(num_gts):
            for i in range(len(num_points_per_lvl)):
                gt_box = gt_bboxes[k]

                down_ratio=self.strides[i]

                trans_output = torch.tensor(
                    [1. / down_ratio, 1. / down_ratio, 0]).to(gt_box.device)
                
                gt_box = affine_transform(gt_box[0:2], trans_output)

                gt_box=gt_box.expand(num_points_per_lvl[i],2)
                if i==0:
                    temp=gt_box
                else:
                    temp=torch.cat((temp,gt_box),dim=0)
            if k==0:
                temp_2=temp.unsqueeze(1)
            else:
                temp_2=torch.cat((temp_2,temp.unsqueeze(1)),dim=1)
        gt_bboxes=temp_2.to(gt_bboxes.device)

        xs = points
        xs = xs[:, None].expand(num_points, num_gts)

        left = xs - gt_bboxes[..., 0]
        right = gt_bboxes[..., 1] - xs
        bbox_targets = torch.stack((left, right), -1)

        # condition1: inside a gt bbox
        inside_gt_bbox_mask = bbox_targets.min(-1)[0] > 0

        # condition2: limit the regression range for each location
        max_regress_distance = bbox_targets.max(-1)[0]
        inside_regress_range = (
            (max_regress_distance >= regress_ranges[..., 0])
            & (max_regress_distance <= regress_ranges[..., 1]))

        # if there are still more than one objects for a location,
        # we choose the one with minimal area
        areas[inside_gt_bbox_mask == 0] = INF
        min_area, min_area_inds = areas.min(dim=1)

        labels = gt_labels[min_area_inds]
        labels[min_area == INF] = 0   # set as BG

        areas[inside_regress_range == 0] = INF
        min_area, min_area_inds = areas.min(dim=1)

        fpn_act_cls_label = gt_labels[min_area_inds]
        fpn_act_cls_label[min_area == INF] = 0
        is_ignore = torch.logical_and((labels > 0), (fpn_act_cls_label == 0))  #0
        labels[is_ignore] = -1

        if self.is_thumos:
            # deal with the 'CliffDiving' and 'Diving' problem
            pos_inds = np.where(labels.cpu() > 0)[0]
            for l in pos_inds:
                single_tmp_in_gt = inside_gt_bbox_mask[l]
                single_tmp_gt_classes = gt_labels[single_tmp_in_gt]
                if 5 in single_tmp_gt_classes and 8 in single_tmp_gt_classes:
                    labels[l] = 5


        bbox_targets = bbox_targets[range(num_points), min_area_inds]

        return labels, bbox_targets

    
    def get_targets(self, points,mlvl_points_concate, gt_bboxes_list, gt_labels_list):
        """Compute regression, classification and centerness targets for points
        in multiple images.

        Args:
            points (list[Tensor]): Points of each fpn level, each has shape
                (num_points, 2).
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image,
                each has shape (num_gt, 4).
            gt_labels_list (list[Tensor]): Ground truth labels of each box,
                each has shape (num_gt,).

        Returns:
            tuple:
                concat_lvl_labels (list[Tensor]): Labels of each level. \
                concat_lvl_bbox_targets (list[Tensor]): BBox targets of each \
                    level.
        """
        assert len(points) == len(self.reg_range_list)
        reg_range_list = self.reg_range_list
        reg_range_list[-1][1]=INF

        num_levels = len(points)
        # expand regress ranges to align with points
        expanded_reg_range_list = [
            mlvl_points_concate[i].new_tensor(reg_range_list[i])[None].expand_as(
                mlvl_points_concate[i]) for i in range(num_levels)
        ]
        # concat all levels points and regress ranges
        concat_reg_range_list = torch.cat(expanded_reg_range_list, dim=0)
        concat_points = torch.cat(points, dim=0)

        # the number of points per img, per lvl
        num_points = [center.size(0) for center in points]

        # get labels and bbox_targets of each image
        labels_list, bbox_targets_list = multi_apply(
            self._get_target_single,
            gt_bboxes_list,
            gt_labels_list,
            points=concat_points,
            regress_ranges=concat_reg_range_list,
            num_points_per_lvl=num_points)

        # concat per level image
        concat_lvl_labels_list = []
        concat_lvl_bbox_targets_list = []

        # split to per img, per level
        for j in range(len(labels_list)):
            concat_lvl_labels_list.append(labels_list[j])
            concat_lvl_bbox_targets_list.append(bbox_targets_list[j])
        return concat_lvl_labels_list, concat_lvl_bbox_targets_list
    
    def loss_single_all_new(self,cls_scores,
            segment_preds,
            labels,
            segment_targets,mlvl_points_concate):
        
        num_imgs = cls_scores[0].size(0) 
        flatten_labels = torch.cat([label for label in labels],0)
        flatten_segment_targets = torch.cat([segment_target for segment_target in segment_targets],0)

        flatten_cls_scores = torch.cat([cls_score for cls_score in cls_scores],2).permute(0,2,1).reshape(-1,self.cls_out_channels)
        flatten_segment_preds = torch.cat([segment_pred for segment_pred in segment_preds],2).permute(0,2,1).reshape(-1,2)

        flatten_points = torch.cat([mlvl_points for mlvl_points in mlvl_points_concate])
        flatten_points = flatten_points.repeat(num_imgs,1)

        pos_inds = (flatten_labels > 0).nonzero().reshape(-1) 

        neg_inds = (flatten_labels == 0).nonzero().reshape(-1)

        segment_weights=torch.zeros(flatten_segment_preds.shape[0],2).to(flatten_labels.device)
        segment_weights[pos_inds]=1

        num_pos = torch.tensor(
                len(pos_inds), dtype=torch.float, device=flatten_labels.device) 
        num_pos = max(reduce_mean(num_pos), 1.0)

        loss_cls = self.loss_cls(flatten_cls_scores, flatten_labels, avg_factor=num_pos)

        pos_points = flatten_points
        pos_decoded_bbox_preds = distance2bbox(pos_points, flatten_segment_preds)
        pos_decoded_target_preds = distance2bbox(pos_points,flatten_segment_targets)

        loss_segment = self.loss_segment(
                    pos_decoded_bbox_preds, 
                    pos_decoded_target_preds, 
                    segment_weights,
                    avg_factor=num_pos)

        return [loss_cls],[loss_segment]


    def loss(self,feats, video_metas, gt_segments,gt_labels, gt_segments_ignore=None): 
        cls_scores, segment_preds=feats
        assert len(cls_scores) == len(segment_preds)

        featmap_sizes = [featmap.size()[-1] for featmap in cls_scores]

        device = cls_scores[0].device
        dtype = cls_scores[0].dtype
        all_level_points,mlvl_points_concate = self.get_points(featmap_sizes, dtype,device)

        
        labels, segment_targets = self.get_targets(all_level_points,mlvl_points_concate, gt_segments,
                                                gt_labels)

        losses_cls, losses_segment=self.loss_single_all_new(cls_scores,segment_preds,labels,segment_targets,mlvl_points_concate)

        loss, log_vars = self._parse_losses(
            dict(loss_cls=losses_cls, loss_segment=losses_segment))

        return dict(loss=loss, log_vars=log_vars)


