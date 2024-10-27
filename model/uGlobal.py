import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch.nn as nn
import math
import warnings
from torchmetrics.functional.classification import auroc
import torch.nn.functional as F
from safetensors.torch import save_file
import pdb
import pickle as pkl
import torchvision
from transformers import get_cosine_schedule_with_warmup

import cv2
import numpy as np
import core.transforms as transforms
import torch.utils.data
import core.checkpoint as checkpoint
from model.CVNet_Rerank_model import CVNet_Rerank
from other.GLAM import GLAM
from torchvision import models
from torch.optim import AdamW

_MEAN = [0.406, 0.456, 0.485]
_SD = [0.225, 0.224, 0.229]

class GeneralizedMeanPooling(nn.Module):

    def __init__(self, norm, output_size=1, eps=1e-6):
        super(GeneralizedMeanPooling, self).__init__()
        assert norm > 0
        self.p = float(norm)
        self.output_size = output_size
        self.eps = eps

    def forward(self, x):
        x = x.clamp(min=self.eps).pow(self.p)
        return torch.nn.functional.adaptive_avg_pool1d(x, self.output_size).pow(1. / self.p)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + str(self.p) + ', ' \
               + 'output_size=' + str(self.output_size) + ')'

class GeneralizedMeanPoolingP(GeneralizedMeanPooling):

    def __init__(self, norm=3, output_size=1, eps=1e-6):
        super(GeneralizedMeanPoolingP, self).__init__(norm, output_size, eps)
        self.p = nn.Parameter(torch.ones(1) * norm)


class GlobalHead(nn.Module):
    def __init__(self, w_in, nc, pp=3):
        super(GlobalHead, self).__init__()
        self.fc = nn.Linear(w_in, nc, bias=True)
        self.pool = GeneralizedMeanPoolingP(norm=pp)
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class sgem(nn.Module):

    def __init__(self, ps=10., infinity = True):
        super(sgem, self).__init__()
        self.ps = ps
        self.infinity = infinity
    def forward(self, x):

        x = torch.stack(x,0)

        if self.infinity:
            x = F.normalize(x, p=2, dim=-1) # 3 C
            x = torch.max(x, 0)[0] 
        else:
            gamma = x.min()
            x = (x - gamma).pow(self.ps).mean(0).pow(1./self.ps) + gamma

        return x
    
class rgem(nn.Module):

    def __init__(self, pr=2.5, size = 5):
        super(rgem, self).__init__()
        self.pr = pr
        self.size = size
        self.lppool = nn.LPPool2d(self.pr, int(self.size), stride=1)
        self.pad = nn.ReflectionPad2d(int((self.size-1)//2.))
    def forward(self, x):
        nominater = (self.size**2) **(1./self.pr)
        x = 0.5*self.lppool(self.pad(x/nominater)) + 0.5*x
        return x

class NonLocalBlock(nn.Module):
    def __init__(self, in_channels):
        super(NonLocalBlock, self).__init__()
        self.in_channels = in_channels
        self.inter_channels = in_channels // 2
        self.g = nn.Conv1d(in_channels, self.inter_channels, kernel_size=1)
        self.theta = nn.Conv1d(in_channels, self.inter_channels, kernel_size=1)
        self.phi = nn.Conv1d(in_channels, self.inter_channels, kernel_size=1)
        self.W = nn.Conv1d(self.inter_channels, in_channels, kernel_size=1)
        self.bn = nn.BatchNorm1d(in_channels)

    def forward(self, x):
        batch_size, C, T = x.size()

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)

        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = self.bn(W_y + x)

        return z
    
class gemp(nn.Module):
    def __init__(self, p=4.6, eps=1e-8, channel=2048, m=2048):
        super(gemp, self).__init__()
        self.m = m
        self.p = [p] * m
        self.eps = [eps] * m
        self.nonlocal_block = NonLocalBlock(channel)
        # self.global_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        pooled_features = []
        for i in range(self.m):
            x1 = self.nonlocal_block(x)
            x_clamped = x1.clamp(self.eps[i]).pow(self.p[i])
            pooled = torch.nn.functional.adaptive_avg_pool1d(x_clamped, 1).pow(1. / self.p[i])
            pooled_features.append(pooled)

        concatenated_features = torch.cat(pooled_features, dim=-1)

        return concatenated_features

class uGlobal(nn.Module):
    def __init__(self, config: dict):
        self.config = config
        self.n_labels = config["n_labels"]
        self.reduction_dim = config["reduction_dim"]
        self.pan = GLAM()
        self.model = config["model"]
        if self.model == "resnet18":
            resnet = models.resnet18(pretrained=True)
            self.resnet50 = torch.nn.Sequential(*list(resnet.children())[:-2])
            for param in self.resnet50.parameters():
                param.requires_grad = False
        elif self.model == "resnet50":
            resnet = models.resnet50(pretrained=True)
            self.resnet50 = torch.nn.Sequential(*list(resnet.children())[:-2])
            for param in self.resnet50.parameters():
                param.requires_grad = False
        elif self.model == "resnet101":
            resnet = models.resnet101(pretrained=True)
            self.resnet = torch.nn.Sequential(*list(resnet.children())[:-2])
            for param in self.resnet.parameters():
                param.requires_grad = False
        else:
            assert "No model availible (Choose [resnet18, resnet50, resnet101])"

        self.gemp = gemp(m=self.reduction_dim)
        self.rgem = rgem()
        self.sgem = sgem()
        self.head - GlobalHead(2048, nc=self.reduction_dim)
        
    def _forward_singlescale(self, x, gemp=True, rgem=True):
        output = self.pan(x)
        output = self.resnet(output)
        if rgem and output.shape[2]>2 and output.shape[3]>2:
            output = self.rgem(output)
        output = output.view(output.shape[0], output.shape[1], -1)
        if gemp:
            output = self.gemp(output)
        else:
            output = self.head.pool(output)
        
        output = torch.transpose(output, 1, 2)
        output = F.normalize(output, p=2, dim=-1)
        
        output = self.head.fc(output)
        
        return output
        
    def forward(self, img, scale=1, labels=None, gemp=True, rgem=True):
        assert scale in [1, 3, 5], "scale must be in [1, 3, 5]"
        feature_list = []
        if scale == 1:
            scale_list = [1.]
        elif scale == 3:
            scale_list = [0.7071, 1., 1.4142]
        elif scale == 5:
            scale_list = [0.5, 0.7071, 1., 1.4142, 2.]
        else:
            raise 
        for _, scl in zip(range(scale), scale_list):
            x = torchvision.transforms.functional.resize(img[0], [int(img[0].shape[-2]*scl),int(img[0].shape[-1]*scl)])
            x = self._forward_singlescale(x, gemp, rgem)
            feature_list.append(x)
        if sgem:
            x_out = self.sgem(feature_list)
        else:
            x_out = torch.stack(feature_list, 0)
            x_out = torch.mean(x_out, 0)

        return x_out