# import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from vedacore.misc import registry,multi_apply
from vedacore.modules import ConvModule, bias_init_with_prob, normal_init

class ScaleLayer(nn.Module):
    
    def __init__(self, init_value=1.0):
        super(ScaleLayer, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale

@registry.register_module('head')
class FcosHead_sigmoid(nn.Module): 
    r"""An anchor-based head used in `RetinaNet
    <https://arxiv.org/pdf/1708.02002.pdf>`_.

    The head contains two subnetworks. The first classifies anchor boxes and
    the second regresses deltas for the anchors.

    Example:
        >>> import torch
        >>> self = RetinaHead(11, 7, 7, conv_cfg=dict(typename='Conv1d'))
        >>> x = torch.rand(1, 7, 32)
        >>> cls_score, seg_pred = self.forward_single(x)
        >>> # Each anchor predicts a score for each class except background
        >>> cls_per_anchor = cls_score.shape[1] / self.num_anchors
        >>> seg_per_anchor = seg_pred.shape[1] / self.num_anchors
        >>> assert cls_per_anchor == (self.num_classes)
        >>> assert seg_per_anchor == 2
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 stacked_convs=4,
                 dcn_on_last_conv=False,
                 feat_channels=256,
                 use_sigmoid=False,
                 num_ins=None,
                 conv_cfg=None,
                 norm_cfg=None,
                 **kwargs):
        super(FcosHead_sigmoid, self).__init__()
        self.stacked_convs = stacked_convs
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.dcn_on_last_conv = dcn_on_last_conv
        self.num_ins=num_ins
        self.in_channels=in_channels
        self.feat_channels=feat_channels
        self.cls_out_channels=num_classes
        self.scales = nn.ModuleList([ScaleLayer(init_value=1.0) for __ in range(self.num_ins)])

        self._init_layers()
        
    def _init_layers(self):
        """Initialize layers of the head."""
        self.relu = nn.ReLU(inplace=False)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            if self.dcn_on_last_conv and i == self.stacked_convs - 1:
                conv_cfg = dict(typename='DCNv2')
            else:
                conv_cfg = self.conv_cfg
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
        self.fcos_cls = nn.Conv1d(
            self.feat_channels,
            self.cls_out_channels,
            3,
            padding=1)
        self.fcos_reg = nn.Conv1d(
            self.feat_channels, 2, 3, padding=1)
        
        self.init_weights()
        


    def init_weights(self):
        """Initialize weights of the head."""
        for m in self.cls_convs:
            normal_init(m.conv, std=0.01)
        for m in self.reg_convs:
            normal_init(m.conv, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.fcos_cls, std=0.01, bias=bias_cls)
        normal_init(self.fcos_reg, std=0.01)

    def forward_single(self, x,scale):
        """Forward feature of a single scale level.

        Args:
            x (Tensor): Features of a single scale level.

        Returns:
            tuple:
                cls_score (Tensor): Cls scores for a single scale level
                    the channels number is num_anchors * num_classes.
                segment_pred (Tensor): Box energies / deltas for a single scale
                    level, the channels number is num_anchors * 2.
        """
        cls_feat = x
        reg_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)
        cls_score = self.fcos_cls(cls_feat)
        segment_pred = self.fcos_reg(reg_feat)
        segment_pred = torch.exp(scale(segment_pred))
        return cls_score, segment_pred
    
    def forward(self, feats):
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 3D-tensor.

        Returns:
            tuple: Usually a tuple of classification scores and segment
                prediction.
                cls_scores (list[Tensor]): Classification scores for all scale
                    levels, each is a 3D-tensor, the channels number is
                    num_anchors * num_classes.
                segment_preds (list[Tensor]): Segment energies / deltas for all
                    scale levels, each is a 3D-tensor, the channels number is
                    num_anchors * 2.
        """
        return multi_apply(self.forward_single, feats,self.scales)
