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
import torch.nn as nn


class residual(nn.Module):
    def __init__(self, in_channel: int, out_channel: int, downsample=None):
        super(residual, self).__init__()
        # self.in_channel = in_channel
        # self.out_channel = out_channel
        self.downsample = downsample
        self.conv1 = nn.Conv2d(in_channel, out_channel // 4, kernel_size=1, padding=0, stride=1)
        self.conv1_rule = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channel // 4, out_channel // 4, kernel_size=3, padding=1, stride=1)
        self.conv3 = nn.Conv2d(out_channel // 4, out_channel, kernel_size=1, padding=0, stride=1)
        self.output_rule = nn.ReLU(inplace=True)

    def forward(self, input_):
        origin = input_
        out = self.conv1_rule(self.conv1(input_))
        out = self.conv2(out)
        out = self.conv3(out)
        if self.downsample:
            origin = self.downsample(origin)
        out += origin
        out = self.output_rule(out)
        return out
