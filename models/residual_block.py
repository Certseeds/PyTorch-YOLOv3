#!/usr/bin/env python3
# coding=utf-8
'''
Github: https://github.com/Certseeds/PyTorch-YOLOv3
Organization: SUSTech
Author: nanoseeds
Date: 2021-03-19 15:16:59
LastEditors: nanoseeds
LastEditTime: 2021-03-19 15:17:50
'''
from typing import List

import torch.nn as nn

from models.residual import residual


class residual_block(nn.Module):
    def __init__(self, in_channel: int, out_channel: int, resNumber: int = 0, downsample=None):
        super(residual_block, self).__init__()
        self.residuals: List[residual] = []
        for i in range(0, resNumber, 1):
            self.residuals.append(residual(in_channel, out_channel, downsample))

    def forward(self, input_):
        origin = input_
        for i in self.residuals:
            x = i(x)
        return x
