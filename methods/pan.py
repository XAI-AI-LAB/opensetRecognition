import time

import torch
import torch.nn as nn
import torch.nn.functional as F

# from .backbone import build_backbone
# from .head import build_head
from methods.neck.builder import build_neck
from methods.conv import Conv_BN_ReLU
from methods.neck.fpem_v1 import FPEM_v1

class PAN(nn.Module):
    def __init__(self, backbone, neck, output_featuremap):
        super(PAN, self).__init__()
        self.backbone = backbone
        self.output_featuremap = output_featuremap
        # in_channels = neck.in_channels
        in_channels = neck['in_channels']
        self.reduce_layer1 = Conv_BN_ReLU(in_channels[0], 128)
        self.reduce_layer2 = Conv_BN_ReLU(in_channels[1], 128)
        self.reduce_layer3 = Conv_BN_ReLU(in_channels[2], 128)
        self.reduce_layer4 = Conv_BN_ReLU(in_channels[3], 128)

        # self.fpem1 = build_neck(neck)
        # self.fpem2 = build_neck(neck)
        self.fpem1 = FPEM_v1(in_channels, neck['out_channels'])
        self.fpem2 = FPEM_v1(in_channels, neck['out_channels'])

        # self.det_head = build_head(detection_head)

    def _upsample(self, x, size, scale=1):
        _, _, H, W = size
        return F.interpolate(x, size=(H // scale, W // scale), mode='bilinear')

    def forward(self,
                imgs,
                gt_texts=None,
                gt_kernels=None,
                training_masks=None,
                gt_instances=None,
                gt_bboxes=None,
                img_metas=None,
                cfg=None):
        outputs = dict()

        # if not self.training and cfg.report_speed:
        torch.cuda.synchronize()
        start = time.time()

        # backbone
        f = self.backbone(imgs)

        # if not self.training and cfg.report_speed:
        torch.cuda.synchronize()
        outputs.update(dict(backbone_time=time.time() - start))
        start = time.time()

        # reduce channel
        f1 = self.reduce_layer1(f[0])
        f2 = self.reduce_layer2(f[1])
        f3 = self.reduce_layer3(f[2])
        f4 = self.reduce_layer4(f[3])

        # FPEM
        f1_1, f2_1, f3_1, f4_1 = self.fpem1(f1, f2, f3, f4)
        f1_2, f2_2, f3_2, f4_2 = self.fpem2(f1_1, f2_1, f3_1, f4_1)

        # FFM
        f1 = f1_1 + f1_2
        f2 = f2_1 + f2_2
        f3 = f3_1 + f3_2
        f4 = f4_1 + f4_2
        
        if self.output_featuremap == 8:
            f2 = self._upsample(f2, f3.size())
            f4 = self._upsample(f4, f3.size())
            f1 = self._upsample(f1, f3.size())
            f = torch.cat((f1, f2, f3, f4), 1)

        elif self.output_featuremap == 4:
            f1 = self._upsample(f1, f4.size())
            f2 = self._upsample(f2, f4.size())
            f3 = self._upsample(f3, f4.size())
            f = torch.cat((f1, f2, f3, f4), 1)

        else:
            raise Exception("output feature map size error")

        # if not self.training and cfg.report_speed:
        torch.cuda.synchronize()
        outputs.update(dict(neck_time=time.time() - start))
        start = time.time()

        return f
        