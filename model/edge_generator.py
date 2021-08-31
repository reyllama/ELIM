import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.bilinear import *
from .sync_batchnorm import SynchronizedBatchNorm2d
import pickle
import numpy as np
from torch.nn import Parameter

class ConvBlock(nn.Module):
    def __init__(self, n_features=256, kernel_size=3, stride=1, padding=1):
        super().__init__()
        
        conv = list()
        conv.append(nn.utils.spectral_norm(nn.Conv2d(n_features, n_features, kernel_size, stride, padding)))
        conv.append(nn.InstanceNorm2d(n_features))
        conv.append(nn.ReLU())
        self.conv = nn.Sequential(*conv)

    def forward(self, x):
        return self.conv(x)

class ResBlock(nn.Module):
    def __init__(self, n_features=256, kernel_size=3, stride=1, padding=1):
        super().__init__()

        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.utils.spectral_norm(nn.Conv2d(n_features, n_features, kernel_size, stride, padding)),
            nn.InstanceNorm2d(n_features),
            nn.ReLU(),
            nn.utils.spectral_norm(nn.Conv2d(n_features, n_features, kernel_size, stride, padding)),
            nn.InstanceNorm2d(n_features),
            nn.ReLU()
        )

        self.upsample = nn.Upsample(scale_factor=2)

    def forward(self, x):
        return self.block(x) + self.upsample(x)

class EdgeGenerator(nn.Module):
    def __init__(self, obj_feat=308, map_size=64):
        # Obj_feat: 128 (z_dim) + 180 (label_dim)
        super(EdgeGenerator, self).__init__()
        self.map_size = map_size

        self.fc = nn.utils.spectral_norm(nn.Linear(obj_feat, 256 * 4 * 4))

        self.conv1 = ResBlock()
        self.conv2 = ResBlock()
        self.conv3 = ResBlock()
        self.conv4 = ResBlock()
        self.conv5 = ResBlock()

        conv6 = list()
        conv6.append(nn.utils.spectral_norm(nn.Conv2d(256, 256, 3, 1, 1)))
        conv6.append(nn.InstanceNorm2d(256))
        conv6.append(nn.ReLU())
        conv6.append(nn.utils.spectral_norm(nn.Conv2d(256, 1, 1, 1))) # 1 channel output from 1x1 conv (no padding)
        conv6.append(nn.Sigmoid())
        self.conv6 = nn.Sequential(*conv5)

    def forward(self, obj_feat, bbox):
        """
        :param obj_feat: (b*num_o, feat_dim)
        :param bbox: (b, num_o, 4)
        :return: bbmap: (b, num_o, map_size, map_size)
        """
        b, num_o, _ = bbox.size()
        obj_feat = obj_feat.view(b * num_o, -1)
        x = self.fc(obj_feat)
        x = self.conv1(x.view(b * num_o, 256, 4, 4)) # 8x8
        # x = F.interpolate(x, size=8, mode='bilinear')
        x = self.conv2(x) # 16x16
        # x = F.interpolate(x, size=16, mode='bilinear')
        x = self.conv3(x) # 32x32
        # x = F.interpolate(x, size=32, mode='bilinear')
        x = self.conv4(x) # 64x64
        # x = F.interpolate(x, size=64, mode='bilinear')
        x = self.conv5(x) # 128x128
        x = self.conv6(x) # 128x128
        edgemap = x.view(b, num_o, 128, 128)

        return edgemap