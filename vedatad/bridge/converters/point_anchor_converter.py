import torch

from vedacore.misc import registry
from vedatad.misc.segment import build_segment_coder
from .base_converter import BaseConverter


def distance2bbox(points, distance, stride, max_shape=None):
    """Decode distance prediction to bounding box.
    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom).
        max_shape (tuple): Shape of the image.
    Returns:
        Tensor: Decoded bboxes.
    """
    x1 = (points - distance[:, 0]) * stride
    x2 = (points + distance[:, 1]) * stride
    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape)
        x2 = x2.clamp(min=0, max=max_shape)
    return torch.stack([x1, x2], -1)


@registry.register_module('converter')
class PointAnchorConverter(BaseConverter):

    def __init__(self, down_ratio_list, num_classes, nms_pre, segment_coder, revert=True, use_sigmoid=True):
        super().__init__()
        self.segment_coder = build_segment_coder(segment_coder)
        self.use_sigmoid_cls = use_sigmoid
        if self.use_sigmoid_cls:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1
        self.nms_pre = nms_pre
        self.revert = revert
        self.down_ratio_list = down_ratio_list

    def get_segments(self, mlvl_points, video_metas, cls_scores,
                     segment_preds):
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
        assert len(cls_scores) == len(segment_preds)
        num_levels = len(cls_scores)

        result_list = []
        for video_id in range(len(video_metas)):
            cls_score_list = [
                cls_scores[i][video_id].detach() for i in range(num_levels)
            ]
            segment_pred_list = [
                segment_preds[i][video_id].detach() for i in range(num_levels)
            ]
            # TODO: hard code. 0 for anchor_list, 1 for valid_flag_list
            # anchors = mlvl_anchors[0][video_id]
            proposals = self._get_segments_single(cls_score_list,
                                                  segment_pred_list, mlvl_points,
                                                  video_metas[video_id],
                                                  self.nms_pre, self.revert)
            result_list.append(proposals)
        return result_list

    def _get_segments_single(self, cls_score_list, segment_pred_list,
                             mlvl_points, video_metas, nms_pre, revert=True):
        """Transform outputs for a single batch item into segment predictions.

        Aapted from https://github.com/open-mmlab/mmdetection

        Args:
            cls_score_list (list[Tensor]): Segment scores for a single scale
                level with shape (num_anchors * num_classes, T).
            segment_pred_list (list[Tensor]): Segment energies / deltas for a
                single scale level with shape (num_anchors * 2, T).
            mlvl_anchors (list[Tensor]): Segment reference for a single scale
                level with shape (num_total_anchors, 2).
            video_metas (dict): Meta information of the video, e.g.,
                tsize, tshift, etc.

        Returns:
            tuple[Tensor, Tensor, Tensor]: The first item is a (n, 2) tensor,
                where 2 columns are time points (start, end). The second item
                is a (n,) tensor where each item is the predicted class score
                of the segment. The third item is a (n,) tensor where each item
                is the predicted centerness score of the segment.
        """
        tsize = video_metas['tsize']
        tscale_factor = video_metas['tscale_factor']
        tshift = video_metas['tshift']

        assert (len(cls_score_list) == len(segment_pred_list) ==
                len(mlvl_points))
        mlvl_segments = []
        mlvl_scores = []
        for cls_score, segment_pred, points, stride in zip(cls_score_list,
                                                           segment_pred_list,
                                                           mlvl_points, self.down_ratio_list):
            assert cls_score.size()[-1] == segment_pred.size()[-1]
            cls_score = cls_score.permute(1,
                                          0).reshape(-1, self.cls_out_channels)
            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                scores = cls_score.softmax(-1)
            segment_pred = segment_pred.permute(1, 0).reshape(-1, 2)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                # Get maximum scores for foreground classes.
                if self.use_sigmoid_cls:
                    max_scores, _ = scores.max(dim=1)
                else:
                    # remind that we set FG labels to [0, num_class-1]
                    # since  v2.0
                    # BG cat_id: num_class
                    max_scores, _ = scores[:, :-1].max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                points = points[topk_inds, :]
                segment_pred = segment_pred[topk_inds, :]
                scores = scores[topk_inds, :]
            segments = distance2bbox(points, segment_pred, stride, max_shape=tsize)
            mlvl_segments.append(segments)
            mlvl_scores.append(scores)
        mlvl_segments = torch.cat(mlvl_segments)
        mlvl_scores = torch.cat(mlvl_scores)
        mlvl_centerness = mlvl_scores.new_ones(mlvl_scores.shape[0]).detach()
        mlvl_segments += tshift
        mlvl_segments /= tscale_factor
        return mlvl_segments, mlvl_scores, mlvl_centerness
