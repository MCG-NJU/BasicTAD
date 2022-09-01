from vedacore.misc import registry
from .infer_engine_AF import InferEngine_AF
import numpy as np


@registry.register_module('engine')
class ValEngine_AF(InferEngine_AF):

    def __init__(self,model, meshgrid, converter, num_classes, use_sigmoid,is_Anet,
                 iou_thr,nmw=False):
        self.nmw=nmw
        super().__init__(model, meshgrid, converter, num_classes, use_sigmoid,is_Anet,
                         iou_thr,nmw)

    def forward(self, data):
        return self.forward_impl(**data)

    def forward_impl(self, imgs, video_metas): 

        dets = self.infer(imgs, video_metas,self.nmw)
        return [dets,len(imgs)]
