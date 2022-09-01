import torch

from vedacore.misc import registry
from vedatad.bridge import build_converter, build_meshgrid
from vedatad.misc.segment import multiclass_nms, segment2result
from .base_engine import BaseEngine


@registry.register_module('engine')
class InferEngine_AF(BaseEngine):

    def __init__(self, model, meshgrid, converter, num_classes, use_sigmoid,is_Anet,
                 iou_thr,nmw=False):
        super().__init__(model)
        self.meshgrid = build_meshgrid(meshgrid)
        self.converter = build_converter(converter)
        if use_sigmoid:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1
        self.iou_thr = iou_thr
        self.is_Anet=is_Anet
        self.nmw=nmw

    def extract_feats(self, img):
        feats = self.model(img, train=False)
        return feats

    def _get_raw_dets(self, imgs, video_metas,iou_thr,nmw):
        """
        Args:
            imgs (torch.Tensor): shape N*3*T*H*W, N is batch size
            video_metas (list): len(video_metas) = N
        Returns:
            dets(list): len(dets) is the batch size, len(dets[ii]) = #classes,
                dets[ii][jj] is an np.array whose shape is N*3
        """
        with torch.no_grad():
            feats = self.extract_feats(imgs)
            dets = self.converter.get_segments(video_metas, feats,iou_thr,nmw)

        return dets

    def _simple_infer(self, imgs, video_metas,nmw):
        """
        Args:
            imgs (torch.Tensor): shape N*3*T*H*W, N is batch size
            video_metas (list): len(video_metas) = N
        Returns:
            dets(list): len(dets) is the batch size, len(dets[ii]) = #classes,
                dets[ii][jj] is an np.array whose shape is N*3
        """
        iou_thr=self.iou_thr
        dets = self._get_raw_dets(imgs, video_metas,iou_thr,nmw)

        result_list = []
        result_list.append(dets)

        return result_list

    def _aug_infer(self, imgs_list, video_metas_list,nmw):
        assert len(imgs_list) == len(video_metas_list)
        dets = []
        ntransforms = len(imgs_list)
        iou_thr=self.iou_thr
        for idx in range(len(imgs_list)):
            imgs = imgs_list[idx]
            video_metas = video_metas_list[idx]
            tdets = self._get_raw_dets(imgs, video_metas,iou_thr,nmw)
            dets.append(tdets)

        return dets

    def infer(self, imgs, video_metas,nmw):
        if len(imgs) == 1:
            if self.is_Anet:
                return self._simple_infer(imgs, video_metas,nmw) 
            else:
                return self._simple_infer(imgs[0], video_metas[0],nmw) 
        else:
            return self._aug_infer(imgs, video_metas,nmw)
