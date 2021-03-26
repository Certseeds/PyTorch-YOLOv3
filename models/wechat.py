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
from __future__ import absolute_import, division, print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.residual import residual

# TODO-add all conv layer to kaiming
# TODO- nn.Sequential replace res_block

from models.residual_block import residual_block


class WeChat(nn.Module):
    def __init__(self):
        # C64
        super(WeChat, self).__init__()
        self.data_bn_scale = nn.BatchNorm2d(3, momentum=0.999, affine=True)
        Batchnorm2d_filler(self.data_bn_scale)
        self.stage1_conv = nn.Conv2d(3, 24, 3, 2, padding=1)
        self.stage1_bn_scale = nn.BatchNorm2d(24, momentum=0.999, affine=True)
        Batchnorm2d_filler(self.stage1_bn_scale)
        self.stage1_relu = nn.ReLU(inplace=True)

        self.stage2 = nn.MaxPool2d(3, 2, 0)

        self.stage3_1_conv = nn.Conv2d(24, 16, 1, 1, padding=0)
        self.stage3_1_conv1_relu = nn.ReLU(inplace=True)
        self.stage3_1_conv2 = nn.Conv2d(16, 16, 3, 2, padding=1)
        self.stage3_1_conv3 = nn.Conv2d(16, 64, 1, 1, padding=0)
        self.stage3_1_conv3_relu = nn.ReLU(inplace=True)
        self.stage3_res_block = residual_block(64, 64, 3)

        self.stage4_1_conv = nn.Conv2d(64, 32, 1, 1, padding=0)
        self.stage4_1_conv1_relu = nn.ReLU(inplace=True)
        self.stage4_1_conv2 = nn.Conv2d(32, 32, 3, 2, padding=1)
        self.stage4_1_conv3 = nn.Conv2d(32, 128, 1, 1, padding=0)
        self.stage4_1_conv3_relu = nn.ReLU(inplace=True)
        self.stage4_res_block = residual_block(128, 128, 7)

        self.stage5_1_conv = nn.Conv2d(128, 32, 1, 1, padding=0)
        self.stage5_1_conv1_relu = nn.ReLU(inplace=True)
        self.stage5_1_conv2 = nn.Conv2d(32, 32, 3, 2, padding=1)
        self.stage5_1_conv3 = nn.Conv2d(32, 128, 1, 1, padding=0)
        self.stage5_1_conv3_relu = nn.ReLU(inplace=True)
        self.stage5_res_block = residual_block(128, 128, 3)

    def forward(self, input_):
        x = self.data_bn_scale(input_)
        x = self.stage1_conv(x)
        x = self.data_bn_scale(x)
        x = self.stage1_relu(x)
        x = self.stage2(x)
        x = self.stage3_1_conv1_relu(self.stage3_1_conv(x))
        x = self.stage3_1_conv2(x)
        x = self.stage3_1_conv3_relu(self.stage3_1_conv(x))
        x = self.stage3_resbolck(x)
        x = self.stage4_1_conv1_relu(self.stage4_1_conv(x))
        x = self.stage4_1_conv2(x)
        x = self.stage4_1_conv3_relu(self.stage4_1_conv(x))
        stage_4_8 = self.stage4_res_block(x)

        stage_5 = self.stage5_1_conv1_relu(self.stage5_1_conv(x))
        stage_5 = self.stage5_1_conv2(stage_5)
        stage_5 = self.stage5_1_conv3_relu(self.stage5_1_conv(stage_5))
        stage_5_4 = self.stage5_res_block(stage_5)



def Batchnorm2d_filler(layer):
    torch.nn.init.constant_(layer.bias, val=0)
    torch.nn.init.constant_(layer.weights, val=1)


# msra_filler == ``
def msra_filler(layer):
    torch.nn.init.kaiming_normal(layer.weights)


class EltWiseSum(nn.Module):
    def __init__(self):
        super(EltWiseSum, self).__init__()

    def forward(self, x, y):
        return torch.add(x, y)
