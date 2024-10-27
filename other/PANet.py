import torch  # 导入PyTorch库
import torch.nn as nn  # 导入PyTorch的神经网络模块
import torch.nn.functional as F  # 导入PyTorch的功能函数模块
import numpy as np  # 导入NumPy库
import other.RepVGG as rv  # 导入自定义的RepVGG模块
import pdb
import gc 
# class PAN(nn.Module):  # 定义PAN类，继承自nn.Module
#     def __init__(self, in_channel_list, out_channel):  # 初始化方法，传入输入通道列表和输出通道数
#         super(PAN, self).__init__()  # 调用父类的初始化方法
#         self.inner_layer1=nn.ModuleList()  # 初始化第一个内部层列表
#         self.inner_layer2=nn.ModuleList()  # 初始化第二个内部层列表
#         self.out_layer1=nn.ModuleList()  # 初始化第一个输出层列表
#         self.out_layer2=nn.ModuleList()  # 初始化第二个输出层列表
#         self.upsp=nn.ModuleList()  # 初始化上采样层列表
#         self.ini=nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=1)  # 定义初始卷积层
#         # self.ini=rv.RepVGGplusBlock(out_channel, out_channel, padding=1)
#         self.col_half=nn.Conv2d(in_channels=out_channel*2, out_channels=out_channel, kernel_size=1)  # 定义卷积层，用于通道减半
#         self.sq = nn.Conv2d(512, out_channels=3, kernel_size=1)

#         for in_channel in in_channel_list:  # 遍历输入通道列表
#             self.inner_layer1.append(nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1))  # 添加第一个内部层
#             self.out_layer1.append(rv.RepVGGplusBlock(out_channel, out_channel, padding=1))  # 添加第一个输出层
#             self.inner_layer2.append(nn.Conv2d(in_channels=out_channel*2, out_channels=out_channel, kernel_size=1))  # 添加第二个内部层
#             self.out_layer2.append(rv.RepVGGplusBlock(out_channel, out_channel, padding=1))  # 添加第二个输出层
#             # self.upsp.append(rv.RepVGGplusBlock(out_channel, out_channel, padding=1))
#             self.upsp.append(nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=1, stride=2))  # 添加上采样层

#     def forward(self, x):  # 前向传播方法
#         head_output=[]  # 初始化头部输出列表
#         panhead_output=[]  # 初始化PAN头部输出列表
#         # a = self.inner_layer1[-1]
#         # b = x[-1]
#         current_inner=self.inner_layer1[-1](x[-1])  # 获取最后一个内部层的输出
#         head_output.append(self.out_layer1[-1](current_inner))  # 将输出添加到头部输出列表中
#         # pdb.set_trace()

#         for i in range(len(x)-2,-1,-1):  # 遍历输入x，从倒数第二个开始
#             pre_inner=current_inner  # 保存当前内部输出
#             current_inner=self.inner_layer1[i](x[i])  # 获取当前内部层的输出
#             size=current_inner.shape[2:]  # 获取当前输出的尺寸
#             top_down=F.interpolate(pre_inner, size=size)  # 对之前的内部输出进行插值
#             cat_pre2current=torch.cat([top_down, current_inner], dim=1)  # 拼接插值后的输出和当前输出
#             col_half=self.col_half(cat_pre2current)  # 通过卷积层
#             head_output.append(self.out_layer1[i](col_half))  # 将输出添加到头部输出列表中

#         # pdb.set_trace()
#         after=self.ini(head_output[len(x)-1])  # 获取头部输出的最后一个元素
#         panhead_output.append(self.out_layer2[len(x)-1](after))  # 将输出添加到PAN头部输出列表中
#         after=self.upsp[len(x)-1](after)  # 通过上采样层



#         # FIXME: 删除了原有的Judge，改为在循环体内使用新的Judge
#         for i in range(len(x)-2,-1,-1):  # 遍历输入x，从倒数第二个开始
#             head_output[i], after = self.Judge2(head_output[i], after)



#             after=torch.cat([head_output[i], after], dim=1)  # 拼接头部输出和上采样后的输出
#             after=self.inner_layer2[i](after)  # 通过内部层
#             panhead_output.append(self.out_layer2[i](after))  # 将输出添加到PAN头部输出列表中
#             after=self.Judge(after)  # 判断尺寸
#             after=self.upsp[i](after)  # 通过上采样层



#         # FIXME: 将不同层的panhead_output的后两维H和W池化成同样大小，并沿着channel维度合并
#         # pdb.set_trace()
#         min_height = panhead_output[len(x)-1].shape[2]
#         min_width = panhead_output[len(x)-1].shape[3]
#         pooled_panhead_output = [F.interpolate(x, size=(min_height, min_width), mode='bilinear', align_corners=False) for x in panhead_output]

#         # pdb.set_trace()
#         final = pooled_panhead_output[len(x)-1]
#         for i in range(len(x)-1):
#             temp = pooled_panhead_output[i]
#             final = torch.cat([final, temp], dim=1)

#         # final = final.to("cpu")
#         # pdb.set_trace()
#         convfinal = self.sq(final)

#         # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         # conv = nn.Conv2d(in_channels=final.shape[1], out_channels=3, kernel_size=1, device=device)
#         # convfinal = conv(final)

#         return convfinal



#     def Judge(self, x):  # 判断方法
#         if x.size()[3] % 2 != 0:  # 如果宽度不是2的倍数
#             x = x[:, :, :, :-1]  # 截断最后一个元素
#         if x.size()[2] % 2 != 0:  # 如果高度不是2的倍数
#             x = x[:, :, :-1, :]  # 截断最后一个元素
#         return x  # 返回结果



#     # FIXME: 新的Judge
#     def Judge2(self, x, y):
#         if x.size()[3] != y.size()[3]:
#             if x.size()[3] > y.size()[3]:
#                 x = x[:, :, :, :-1]
#             else:
#                 y = y[:, :, :, :-1]
#         if x.size()[2] != y.size()[2]:
#             if x.size()[2] > y.size()[2]:
#                 x = x[:, :, :-1, :]
#             else:
#                 y = y[:, :, :-1, :]
#         return x, y

import pdb
class PAN(nn.Module):
    def __init__(self, in_channel_list, out_channel):
        super(PAN, self).__init__()
        self.out_channel = out_channel
        self.num_levels = len(in_channel_list)

        # 使用nn.ModuleList替代多个独立的ModuleList
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'inner1': nn.Conv2d(in_channel, out_channel, kernel_size=1),
                'out1': rv.RepVGGplusBlock(out_channel, out_channel, padding=1),
                'inner2': nn.Conv2d(out_channel, out_channel, kernel_size=1),
                'out2': rv.RepVGGplusBlock(out_channel, out_channel, padding=1),
                'upsp': nn.Conv2d(out_channel, out_channel, kernel_size=1, stride=2)
            }) for in_channel in in_channel_list
        ])

        self.ini = nn.Conv2d(out_channel, out_channel, kernel_size=1)
        self.col_half = nn.Conv2d(out_channel, out_channel, kernel_size=1)
        self.sq = nn.Conv2d(128, 3, kernel_size=1)

    def forward(self, x):
        # pdb.set_trace()

        assert len(x) == self.num_levels, "Input list length must match the number of levels"

        # FIXME 改了
        head_output = [None] * self.num_levels # 预先用 None 初始化列表
        panhead_output = [None] * self.num_levels  # 预先用 None 初始化列表
        # head_output = []
        # panhead_output = []

        # 自顶向下的路径
        current_inner = self.layers[-1]['inner1'](x[-1])
        # head_output.append(self.layers[-1]['out1'](current_inner))
        head_output[0] = self.layers[-1]['out1'](current_inner)

        #10G 存款
        for i in range(self.num_levels - 2, -1, -1):
            pre_inner = current_inner
            current_inner = self.layers[i]['inner1'](x[i])
            size = current_inner.shape[2:]
            top_down = F.interpolate(pre_inner, size=size, mode='nearest')
            # cat_pre2current = torch.cat([top_down, current_inner], dim=1)
            cat_pre2current = torch.mean(torch.stack([top_down, current_inner], dim=0), dim=0)
            col_half = self.col_half(cat_pre2current)
            # head_output.append(self.layers[i]['out1'](col_half))
            # FIXME 改了
            head_output[3-i] = self.layers[i]['out1'](col_half)

        # 自底向上的路径
        after = self.ini(head_output[-1])
        # panhead_output.append(self.layers[-1]['out2'](after))
        # FIXME 改了
        panhead_output[0] = self.layers[-1]['out2'](after)
        after = self.layers[-1]['upsp'](after)

        for i in range(self.num_levels - 2, -1, -1):
            head_output[i], after = self.align_features(head_output[i], after)
            # after = torch.cat([head_output[i], after], dim=1)
            after = torch.mean(torch.stack([head_output[i], after], dim=0), dim=0)
            after = self.layers[i]['inner2'](after)
            # panhead_output.append(self.layers[i]['out2'](after))
            # FIXME 改了
            panhead_output[3-i] = self.layers[i]['out2'](after)
            after = self.align_features(after, after)[0]
            after = self.layers[i]['upsp'](after)

        # # # 特征融合
        # min_size = panhead_output[-1].shape[2:]
        # pooled_panhead_output = [F.interpolate(x, size=min_size, mode='bilinear', align_corners=False) for x in panhead_output]
        # # final = torch.cat(pooled_panhead_output, dim=1)
        # final = torch.mean(torch.stack(pooled_panhead_output, dim=0), dim=0)
        # # print(torch.cuda.memory_allocated(device=0)/1e9)

        # pdb.set_trace()

        # # FIXME: 0806pm: 将池化为最小改为池化为最大
        max_size = panhead_output[0].shape[2:]
        pooled_panhead_output = [F.interpolate(x, size=max_size, mode='bilinear', align_corners=False) for x in panhead_output]
        # final = torch.cat(pooled_panhead_output, dim=1)
        final = torch.mean(torch.stack(pooled_panhead_output, dim=0), dim=0)


        return self.sq(final)


    @staticmethod
    def align_features(x, y):
        min_height = min(x.size(2), y.size(2))
        min_width = min(x.size(3), y.size(3))
        x = x[:, :, :min_height, :min_width]
        y = y[:, :, :min_height, :min_width]
        return x, y
