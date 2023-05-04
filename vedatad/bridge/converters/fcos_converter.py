import torch

from vedacore.misc import registry
from vedatad.misc.segment import build_segment_coder
from .base_converter import BaseConverter
import time
import numpy as np

def segment_overlaps(segments1,
                     segments2,
                     mode='iou',
                     is_aligned=False,
                     eps=1e-6):
    is_numpy = False
    if isinstance(segments1, np.ndarray):
        segments1 = torch.from_numpy(segments1)
        is_numpy = True
    if isinstance(segments2, np.ndarray):
        segments2 = torch.from_numpy(segments2)
        is_numpy = True

    assert mode in ['iou', 'iof']
    # Either the segments are empty or the length of segments's last dimenstion
    # is 2
    assert (segments1.size(-1) == 2 or segments1.size(0) == 0)
    assert (segments2.size(-1) == 2 or segments2.size(0) == 0)

    rows = segments1.size(0)
    cols = segments2.size(0)
    if is_aligned:
        assert rows == cols

    if rows * cols == 0:
        return segments1.new(rows, 1) if is_aligned else segments2.new(
            rows, cols)

    if is_aligned:
        start = torch.max(segments1[:, 0], segments2[:, 0])  # [rows]
        end = torch.min(segments1[:, 1], segments2[:, 1])  # [rows]

        overlap = (end - start).clamp(min=0)  # [rows, 2]
        area1 = segments1[:, 1] - segments1[:, 0]

        if mode == 'iou':
            area2 = segments2[:, 1] - segments2[:, 0]
            union = area1 + area2 - overlap
        else:
            union = area1
    else:
        start = torch.max(segments1[:, None, 0], segments2[:,
                                                           0])  # [rows, cols]
        end = torch.min(segments1[:, None, 1], segments2[:, 1])  # [rows, cols]

        overlap = (end - start).clamp(min=0)  # [rows, cols]
        area1 = segments1[:, 1] - segments1[:, 0]

        if mode == 'iou':
            area2 = segments2[:, 1] - segments2[:, 0]
            union = area1[:, None] + area2 - overlap
        else:
            union = area1[:, None]

    eps = union.new_tensor([eps])
    union = torch.max(union, eps)
    ious = overlap / union

    if is_numpy:
        ious = ious.numpy()

    return ious

def fpn_singlecls_decode(act_cls_list, det_bbox_list, act_loss_type,down_ratio_list,norm_offset_reg):
    num_lvls = len(down_ratio_list)
    
    output_act_cls = []
    output_det_bboxes = []
    for i in range(num_lvls):
        down_ratio = down_ratio_list[i]
        act_cls = act_cls_list[i]  
        det_bbox = det_bbox_list[i]  # NTx2
        batchsize, num_classes, output_length = act_cls.shape[:3]
        act_cls = act_cls.permute(0, 2, 1)  # NxTxC
        det_bbox = det_bbox.reshape(batchsize, output_length, 2)  # NxTx2
        if act_loss_type == 'softmax' or act_loss_type == 'focal' or act_loss_type == 'focal_single':
            # background plus C act classes
            act_cls = act_cls[:, :, 1:]
        
        # clamp out of boundary bboxes
        det_bbox = det_bbox.clamp_(min=0.0, max=output_length*down_ratio)  # N*T*2

        output_act_cls.append(act_cls)
        output_det_bboxes.append(det_bbox)
    
    output_act_cls = torch.cat(output_act_cls, dim=1)
    output_det_bboxes = torch.cat(output_det_bboxes, dim=1)

    return output_act_cls, output_det_bboxes

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

def fpn_decode(act_cls_list, offset_reg_list, act_loss_type,down_ratio_list,norm_offset_reg):
    num_lvls = len(down_ratio_list)

    output_act_cls = []
    output_det_bboxes = []
    for i in range(num_lvls):
        act_cls = act_cls_list[i]
        offset_reg = offset_reg_list[i]
        down_ratio = down_ratio_list[i]

        batchsize, num_classes, output_length = act_cls.shape[:3]
        
        act_cls = act_cls.permute(0, 2, 1)  # N*T*C
        if act_loss_type == 'softmax' or act_loss_type == 'focal':
            # background plus C act classes
            act_cls = act_cls[:, :, 1:]
            
        offset_reg = offset_reg.permute(0, 2, 1)  # N*T*2

        if norm_offset_reg:
            tmp_locations = torch.arange(0, output_length, dtype=offset_reg.dtype,
                                         device=offset_reg.device) + 0.5
        else:
            tmp_locations = torch.arange(0, output_length*down_ratio, step=down_ratio,
                                         dtype=offset_reg.dtype, device=offset_reg.device) \
                            + 0.5 * down_ratio
        
        tmp_locations = tmp_locations.repeat(batchsize, 1)  # N*T
        det_bboxes = torch.zeros_like(offset_reg)  # N*T*2
        det_bboxes[:, :, 0] = tmp_locations - offset_reg[:, :, 0]
        det_bboxes[:, :, 1] = offset_reg[:, :, 1] + tmp_locations

        # map det_bboxes to original length
        det_bboxes = det_bboxes * down_ratio
        # clamp out of boundary bboxes
        det_bboxes = det_bboxes.clamp_(min=0.0, max=output_length*down_ratio)  # N*T*2

        output_act_cls.append(act_cls)
        output_det_bboxes.append(det_bboxes)

    output_act_cls = torch.cat(output_act_cls, dim=1)
    output_det_bboxes = torch.cat(output_det_bboxes, dim=1)

    return output_act_cls, output_det_bboxes

def fpn_singlecls_decode(act_cls_list, det_bbox_list, act_loss_type,down_ratio_list,norm_offset_reg):
    num_lvls = len(down_ratio_list)
    
    output_act_cls = []
    output_det_bboxes = []
    for i in range(num_lvls):
        down_ratio = down_ratio_list[i]
        act_cls = act_cls_list[i]  
        det_bbox = det_bbox_list[i]  # NTx2
        batchsize, num_classes, output_length = act_cls.shape[:3]
        act_cls = act_cls.permute(0, 2, 1)  # NxTxC
        det_bbox = det_bbox.reshape(batchsize, output_length, 2)  # NxTx2
        if act_loss_type == 'softmax' or act_loss_type == 'focal' or act_loss_type == 'focal_single':
            # background plus C act classes
            act_cls = act_cls[:, :, 1:]
        
        # clamp out of boundary bboxes
        det_bbox = det_bbox.clamp_(min=0.0, max=output_length*down_ratio)  # N*T*2

        output_act_cls.append(act_cls)
        output_det_bboxes.append(det_bbox)
    
    output_act_cls = torch.cat(output_act_cls, dim=1)
    output_det_bboxes = torch.cat(output_det_bboxes, dim=1)

    return output_act_cls, output_det_bboxes

def bbox_transform_inv(roi_bbox, roi_reg):
    # import pdb; pdb.set_trace()
    # roi_bbox: (numroi, 2)
    # roi_reg: (numroi, 2)
    lens = roi_bbox[:, 1] - roi_bbox[:, 0] + 1.0 
    ctr_x = roi_bbox[:, 0] + 0.5 * lens 
    # after the slice operation, dx will be (numroi, 1), Not (numroi,)
    dx = roi_reg[:, 0::2] #delta^x
    dl = roi_reg[:, 1::2] #delta^w

    pred_ctr_x = dx * lens.unsqueeze(1) + ctr_x.unsqueeze(1) 
    pred_l = torch.exp(dl) * lens.unsqueeze(1) #w^2

    pred_rois = roi_reg.clone()
    pred_rois[:, 0::2] = pred_ctr_x - 0.5 * pred_l
    pred_rois[:, 1::2] = pred_ctr_x + 0.5 * pred_l

    return pred_rois

def nms(dets, thresh=0.4):
    """Pure Python NMS baseline."""
    if len(dets) == 0:
        return []

    start = dets[:, 0]
    end = dets[:, 1]
    scores = dets[:, 2]
    lengths = end - start

    order = scores.argsort(descending=True)
    keep = []
    while order.size()[0] > 0:
        i = order[0]
        keep.append(int(i))
        rest_start = torch.max(start[i], start[order[1:]])
        rest_end = torch.min(end[i], end[order[1:]])
        inter = (rest_end - rest_start).clamp(min=0.0)

        ious = inter / (lengths[i] + lengths[order[1:]] - inter)
        inds = ious <= thresh
        order = order[1:][inds]
    return torch.tensor(keep)

@registry.register_module('converter')
class FcosConverter(BaseConverter):

    def __init__(self, model,act_loss_type,down_ratio_list,act_score_thresh,norm_offset_reg,max_proposal,is_Anet):
        super().__init__()
        self.model=model
        self.act_loss_type=act_loss_type
        self.down_ratio_list=down_ratio_list
        self.act_score_thresh=act_score_thresh
        self.norm_offset_reg=norm_offset_reg
        self.max_proposal=max_proposal
        self.is_Anet=is_Anet
    
    def process(self, model_output, return_time=False):
            num_lvls = len(model_output)

            act_cls_list = []
            offset_reg_list = []

            for i in range(num_lvls):
                # import pdb; pdb.set_trace()
                output = model_output[i]

                if self.act_loss_type == 'softmax' or self.act_loss_type == 'focal':
                    # N*C*T
                    act_cls = torch.nn.functional.softmax(output['act_cls'], dim=1)
                else:
                    act_cls = torch.sigmoid(output['act_cls'])


                offset_reg = output['offset_reg']  # N*2*T
                offset_reg_list.append(offset_reg)

                act_cls_list.append(act_cls)  # N*C*T


            
            forward_time = time.time()
            # transform the offsets to [start, end] boxes.
            # map to the original length, 768
            # scores: N*T*20
            # dets: N*T*2
            # cls from 0 to 19
            scores, det_bboxes = fpn_decode(act_cls_list, offset_reg_list, self.act_loss_type,self.down_ratio_list,self.norm_offset_reg)

            if return_time:
                return model_output, (scores, det_bboxes), forward_time
            else:
                return model_output, (scores, det_bboxes) 
    def process_anet(self, model_output, return_time=False):
            num_lvls = len(model_output)

            act_cls_list = []
            offset_reg_list = []
            det_bbox_list = []

            for i in range(num_lvls):
                output = model_output[i]
                if self.act_loss_type == 'softmax' or self.act_loss_type == 'focal' or self.act_loss_type == 'focal_single':
                    # N*C*T
                    act_cls = torch.nn.functional.softmax(output['act_cls'], dim=1)
                else:
                    act_cls = torch.sigmoid(output['act_cls'])

                offset_reg = output['offset_reg']  # N*2*T
                offset_reg = offset_reg.permute(0, 2, 1)  # N*T*2
                
                # NT * C
                combined_cls = torch.cat([act_cls], dim=2)

                act_cls_list.append(combined_cls)  

                roi_bbox = offsets2bboxes(offset_reg, self.down_ratio_list[i])  # NT*2
                roi_bbox = roi_bbox.view(-1, 2)
                det_bbox_list.append(roi_bbox)

            
            forward_time = time.time()
            scores, det_bboxes = fpn_singlecls_decode(act_cls_list, det_bbox_list, self.act_loss_type,self.down_ratio_list,self.norm_offset_reg)

            if return_time:
                return model_output, (scores, det_bboxes), forward_time
            else:
                return model_output, (scores, det_bboxes) 
    def post_process(self, dets, meta_list,iou_thr):
        # import pdb; pdb.set_trace()
        scores = dets[0]  # N*T*C
        det_bboxes = dets[1]  # N*T*2
        batchsize = scores.shape[0]
        num_classes = scores.shape[2]
        assert scores.shape[0] == det_bboxes.shape[0]

        top_preds_list = []
        thresh = self.act_score_thresh
        for b in range(batchsize):

            start_frame = float(meta_list[b]['tshift'])
            skip_frame = 1.0
            single_score = scores[b]  
            single_bboxes = det_bboxes[b]  # T*2
            top_preds = {}

            all_scores = np.zeros((0,))
            for j in range(num_classes):
                # import pdb; pdb.set_trace()
                inds = torch.nonzero(single_score[:, j] > thresh).reshape(-1)
                if inds.numel() > 0:
                    cls_scores = single_score[inds, j]  # M
                    cls_bboxes = single_bboxes[inds, :]  # M*2

                    cls_dets = torch.cat((cls_bboxes, cls_scores.reshape(-1,1)), dim=1)  # M*3
                    # nms for per-class predictions
                    keep = nms(cls_dets, iou_thr)  
                    cls_dets = cls_dets[keep].cpu().numpy()
                    cls_dets[:, :2] = cls_dets[:, :2] * skip_frame + start_frame
                    top_preds[j+1] = cls_dets
                    # import pdb; pdb.set_trace()
                    all_scores = np.hstack((all_scores, cls_dets[:, 2]))
                else:
                    top_preds[j+1] = np.zeros((0, 3))

            # limit to 200 detections over all classes
            # import pdb; pdb.set_trace()
            if all_scores.shape[0] > self.max_proposal:
                # import pdb; pdb.set_trace()
                score_thresh = np.sort(all_scores)[-self.max_proposal]
                for j in range(num_classes):
                    keep = np.where(top_preds[j+1][:, 2] >= score_thresh)[0]
                    top_preds[j+1] = top_preds[j+1][keep, :]

            top_preds_list.append(top_preds)

        return top_preds_list
    
    def post_process_nmw(self, dets, meta_list,iou_thr):
        def _nmw(segments, scores, labels, keep):
            mask = labels[keep] == labels[keep][0]
            segments = segments[keep][mask]
            scores = scores[keep][mask]
            labels = labels[keep][mask]

            ious = segment_overlaps(segments[:1], segments, mode='iou')[0]
            ious[0] = 1.0
            iou_mask = ious >= iou_thr
            accu_segments = segments[iou_mask]
            accu_weights = scores[iou_mask] * ious[iou_mask]
            accu_weights /= accu_weights.sum()
            segment = (accu_weights[:, None] * accu_segments).sum(dim=0)
            score = scores[0]
            label = labels[0]

            inds = torch.nonzero(mask)[:, 0]
            mask[inds[~iou_mask]] = False
            keep = keep[~mask]

            return segment, score, label, keep

        # import pdb; pdb.set_trace()
        scores = dets[0]  # N*T*C
        det_bboxes = dets[1]  # N*T*2 
        batchsize = scores.shape[0]
        num_classes = scores.shape[2]
        assert scores.shape[0] == det_bboxes.shape[0]

        top_preds_list = []
        thresh = self.act_score_thresh
        for b in range(batchsize):

            start_frame = float(meta_list[b]['tshift'])
            skip_frame = 1.0
            single_score = scores[b]  
            single_bboxes = det_bboxes[b]  # T*2 
            top_preds = {}

            #nmw
            for  i in range(single_score.shape[1]):
                top_preds[i+1]=[]
            num_classes = single_score.size(1) #20
            segments = single_bboxes[:, None].expand(-1, num_classes, 2)
            scores = single_score[:, :]
            valid_mask = scores > 0.005 
            segments = segments[valid_mask]
            scores = scores[valid_mask]

            labels = valid_mask.nonzero(as_tuple=False)[:, 1]
            keep = scores.argsort(descending=True)
            results = []
            max_num = self.max_proposal

            while keep.numel() > 0:
                segment, score, label, keep = _nmw(segments, scores, labels, keep)
                segment=segment* skip_frame + start_frame

                top_preds[label.item()+1].append(torch.cat([segment,score[None]],dim=-1).cpu().numpy())
                if max_num > 0 and len(results) == max_num:
                    break
            for j in range(num_classes):
                if top_preds[j+1]:
                    temp=np.stack(top_preds[j+1])
                    top_preds[j+1]=torch.from_numpy(temp)
                else:
                    top_preds[j+1]=np.array(top_preds[j+1])

            top_preds_list.append(top_preds)

        return top_preds_list

    def post_process_anet(self, dets, meta_list,iou_thr):
        scores = dets[0]  # N*T*C

        det_bboxes = dets[1]  # N*T*2
        batchsize = scores.shape[0]
        num_classes = scores.shape[2]
        assert scores.shape[0] == det_bboxes.shape[0]

        top_preds_list = []
        thresh = self.act_score_thresh
        # map the temporal window to the original input videos
        for b in range(batchsize):
            skip_frame = float(meta_list[b]['step'])
            resize = float(meta_list[b]['resize'])

            single_score = scores[b]  
            single_bboxes = det_bboxes[b]  # T*2
            
            top_preds = {}

            all_scores = np.zeros((0,))
            for j in range(num_classes):
                inds = torch.nonzero(single_score[:, j] > thresh).reshape(-1)
                if inds.numel() > 0:
                    cls_scores = single_score[inds, j]  
                    cls_bboxes = single_bboxes[inds, :]  

                    cls_dets = torch.cat((cls_bboxes, cls_scores.reshape(-1,1)), dim=1)  
                    
                    # nms for per-class predictions
                    keep = nms(cls_dets, iou_thr)  
                    cls_dets = cls_dets[keep].cpu().numpy()
                    cls_dets[:, :2] = cls_dets[:, :2] * skip_frame * resize 
                    top_preds[j+1] = cls_dets
                    all_scores = np.hstack((all_scores, cls_dets[:, 2]))
                else:
                    top_preds[j+1] = np.zeros((0, 3))

            # limit to 200 detections over all classes
            if all_scores.shape[0] > 200:
                score_thresh = np.sort(all_scores)[-200]
                for j in range(num_classes):
                    keep = np.where(top_preds[j+1][:, 2] >= score_thresh)[0]
                    top_preds[j+1] = top_preds[j+1][keep, :]

            top_preds_list.append(top_preds)

        return top_preds_list

    def get_segments(self, video_metas,feats,iou_thr,nmw):
        """Transform network output for a batch into segment predictions.

        Aapted from https://github.com/open-mmlab/mmdetection

        Args:
            video_metas (list[dict]): Meta information of each video, e.g.,
                tsize, tshift, etc.
            cls_scores (list[Tensor]): Segment scores for each scale level
                Has shape (N, num_anchors * num_classes, T)
            segment_preds (list[Tensor]): Segment energies / deltas for each
                scale level with shape (N, num_anchors * 2, T)

        Returns:
            list[tuple[Tensor, Tensor, Tensor]]: Each item in result_list is
                3-tuple. The first item is a (n, 2) tensor, where 2 columns are
                time points (start, end). The second item is a (n,) tensor
                where each item is the predicted class score of the segment.
                The third item is a (n,) tensor where each item is the
                predicted centerness score of the segment.

        Example:
        """
        video_blob = feats
        meta_list = video_metas
        torch.cuda.synchronize()

        # output cls scores and offsets
        # map the obtained windows to original video length
        if self.is_Anet:
            __, dets, forward_time = self.process_anet(video_blob, return_time=True) 
        else:
            __, dets, forward_time = self.process(video_blob, return_time=True)

        torch.cuda.synchronize()

        # apply nms per cls on det rsts
        if self.is_Anet:             
            results_list = self.post_process_anet(dets, meta_list,iou_thr)
        else:
            if nmw==True:
                results_list = self.post_process_nmw(dets, meta_list,iou_thr)
            else:
                results_list = self.post_process(dets, meta_list,iou_thr)
        torch.cuda.synchronize()

        return {
            'results': results_list,
        }