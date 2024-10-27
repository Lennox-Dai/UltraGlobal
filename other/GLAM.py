# RepVGG & PAN
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import other.RepVGG as rv
import other.PANet
import pdb

class GLAM(nn.Module):  # 定义GLAM类，继承自nn.Module
    def __init__(self, shape = (26,57), **kwargs):  # 初始化方法，传入形状参数和其他关键字参数
        super(GLAM, self).__init__()  # 调用父类的初始化方法
        self.inchannels = []  # 初始化输入通道列表
        self.outchannels = 256  # 设置输出通道数



        # FIXME: 更改了in_channels
        self.conva1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(1, 3), stride=1, padding=(0, 1))  # 定义卷积层
        self.batchnormb1 = nn.BatchNorm2d(16)  # 定义批归一化层
        self.conva2 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 1), stride=1, padding=(1, 0))  # 定义卷积层



        self.batchnormb2 = nn.BatchNorm2d(16)  # 定义批归一化层
        self.MSB1 = rv.RepVGGplusBlock(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)  # 定义RepVGGplus块
        self.maxp1 = nn.MaxPool2d(kernel_size=(2, 2))  # 定义最大池化层
        self.MSB2 = rv.RepVGGplusBlock(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)  # 定义RepVGGplus块
        self.maxp2 = nn.MaxPool2d(kernel_size=(2, 2))  # 定义最大池化层
        self.MSB3 = rv.RepVGGplusBlock(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)  # 定义RepVGGplus块
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(5, 5), stride=1, padding=2)  # 定义卷积层
        self.maxp3 = nn.MaxPool2d(kernel_size=(2, 2))  # 定义最大池化层
        self.bn = nn.BatchNorm2d(128)  # 定义批归一化层



        # FIXME: 去除了将张量后两维合并的操作所需对象的定义
        # dim = (shape[0]*2) * (shape[1]) + (shape[0]) * (shape[1]//2) + (shape[0]//2) * (shape[1]//4) + (shape[0]//4) * (shape[1]//8)  # 计算维度
        # self.gmlp = gMLP(dim=dim, depth=1, seq_len=128, act=nn.Tanh())  # 定义gMLP层
        # self.fc = nn.Linear(in_features=dim*128, out_features=4)  # 定义全连接层



        self.pan = other.PANet.PAN([16, 32, 64, 128], 128)  # 定义PANet层
        self.pan = self.pan.cuda()

    def forward(self, *input):  # 前向传播方法
        # pdb.set_trace()
        self.features = []  # 初始化特征列表
        # 输入数据的大小：[32, 1, 26, 57]
        xa = self.conva1(input[0])  # 通过第一层卷积
        xa = self.batchnormb1(xa)  # 通过批归一化

        xb = self.conva2(input[0])  # 通过第二层卷积
        xb = self.batchnormb2(xb)  # 通过批归一化

        xa = F.relu(xa)  # 通过ReLU激活函数
        xb = F.relu(xb)  # 通过ReLU激活函数

        x = torch.cat([xa, xb], dim = 2)  # 拼接两个卷积输出
        self.features.append(x)  # 将拼接结果添加到特征列表中
        self.inchannels.append(x.size()[1])  # 将输入通道数添加到输入通道列表中

        x = self.MSB1(x)  # 通过第一个RepVGGplus块
        x = self.maxp1(x)  # 通过最大池化层
        self.features.append(x)  # 将结果添加到特征列表中
        self.inchannels.append(x.size()[1])  # 将输入通道数添加到输入通道列表中

        x = self.MSB2(x)  # 通过第二个RepVGGplus块
        x = self.maxp2(x)  # 通过最大池化层
        self.features.append(x)  # 将结果添加到特征列表中
        self.inchannels.append(x.size()[1])  # 将输入通道数添加到输入通道列表中

        x = self.MSB3(x)  # 通过第三个RepVGGplus块
        x = self.conv3(x)  # 通过卷积层
        x = self.maxp3(x)  # 通过最大池化层
        x = self.bn(x)  # 通过批归一化层

        x = F.relu(x)  # 通过ReLU激活函数
        self.outchannels = x.size()[1]  # 获取输出通道数
        self.features.append(x)  # 将结果添加到特征列表中
        self.inchannels.append(x.size()[1])  # 将输入通道数添加到输入通道列表中
        # print(x.shape)

        # print(">>>>>>>>>>>>>>>>self.features")
        # print(len(self.features))
        # print(len(self.features[0]))
        # print(self.features[0].shape)
        # pdb.set_trace()
        
        x = self.pan(self.features)  # 通过PANet层



        # FIXME: 去除了将张量后两维合并的操作
        # x = self.gmlp(x)  # 通过gMLP层
        # x = x.reshape(x.shape[0], -1)  # 重塑张量形状
        # x = self.fc(x)  # 通过全连接层



        return x  # 返回结果
