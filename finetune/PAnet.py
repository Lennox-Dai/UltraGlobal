import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

class CustomConvModule(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size, 
                 stride=1, 
                 padding=0, 
                 dilation=1, 
                 groups=1, 
                 bias=True,
                 norm_type='bn',
                 act_type='relu',
                 order=('conv', 'norm', 'act')):
        super(CustomConvModule, self).__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, 
                              stride, padding, dilation, groups, bias)
        
        # Normalization layer
        if norm_type == 'bn':
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm_type == 'in':
            self.norm = nn.InstanceNorm2d(out_channels)
        elif norm_type is None:
            self.norm = nn.Identity()
        else:
            raise ValueError(f"Unsupported normalization type: {norm_type}")
        
        # Activation function
        if act_type == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif act_type == 'leaky_relu':
            self.act = nn.LeakyReLU(inplace=True)
        elif act_type is None:
            self.act = nn.Identity()
        else:
            raise ValueError(f"Unsupported activation type: {act_type}")
        
        self.order = order

    def forward(self, x):
        for layer in self.order:
            if layer == 'conv':
                x = self.conv(x)
            elif layer == 'norm':
                x = self.norm(x)
            elif layer == 'act':
                x = self.act(x)
        return x
    
class PANet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, feature_channels=64):
        super(PANet, self).__init__()
        
        # Bottom-up pathway (modified ResNet-like structure)
        self.c1 = CustomConvModule(in_channels, feature_channels, kernel_size=7, stride=2, padding=3)
        self.c2 = CustomConvModule(feature_channels, feature_channels * 2, 3, stride=2)
        self.c3 = CustomConvModule(feature_channels * 2, feature_channels * 4, 4, stride=2)
        self.c4 = CustomConvModule(feature_channels * 4, feature_channels * 8, 6, stride=2)
        self.c5 = CustomConvModule(feature_channels * 8, feature_channels * 16, 3, stride=2)

        # Lateral connections
        self.lateral_c5 = CustomConvModule(feature_channels * 16, feature_channels, kernel_size=1)
        self.lateral_c4 = CustomConvModule(feature_channels * 8, feature_channels, kernel_size=1)
        self.lateral_c3 = CustomConvModule(feature_channels * 4, feature_channels, kernel_size=1)
        self.lateral_c2 = CustomConvModule(feature_channels * 2, feature_channels, kernel_size=1)

        # Top-down pathway
        self.fpn_p5 = CustomConvModule(feature_channels, feature_channels, kernel_size=3, padding=1)
        self.fpn_p4 = CustomConvModule(feature_channels, feature_channels, kernel_size=3, padding=1)
        self.fpn_p3 = CustomConvModule(feature_channels, feature_channels, kernel_size=3, padding=1)
        self.fpn_p2 = CustomConvModule(feature_channels, feature_channels, kernel_size=3, padding=1)

        # Bottom-up path in PANet
        self.bu_p2 = CustomConvModule(feature_channels, feature_channels, kernel_size=3, stride=2, padding=1)
        self.bu_p3 = CustomConvModule(feature_channels, feature_channels, kernel_size=3, stride=2, padding=1)
        self.bu_p4 = CustomConvModule(feature_channels, feature_channels, kernel_size=3, stride=2, padding=1)

        # Final layers
        self.final_conv = CustomConvModule(feature_channels * 4, out_channels, kernel_size=3, padding=1)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(CustomConvModule(in_channels, out_channels, kernel_size=3, stride=stride, padding=1))
        for _ in range(1, num_blocks):
            layers.append(CustomConvModule(out_channels, out_channels, kernel_size=3, stride=1, padding=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        # Bottom-up pathway
        # pdb.set_trace()
        h = x.shape[2]
        w = x.shape[3]
        # x = torch.zeros(1, 3, 5, 7).to('cuda')
        c1 = self.c1(x)
        c2 = self.c2(c1)
        c3 = self.c3(c2)
        c4 = self.c4(c3)
        c5 = self.c5(c4)

        # Top-down pathway
        p5 = self.fpn_p5(self.lateral_c5(c5))
        p4 = self.fpn_p4(self.lateral_c4(c4) + F.interpolate(p5, size=c4.shape[2:], mode='nearest'))
        p3 = self.fpn_p3(self.lateral_c3(c3) + F.interpolate(p4, size=c3.shape[2:], mode='nearest'))
        p2 = self.fpn_p2(self.lateral_c2(c2) + F.interpolate(p3, size=c2.shape[2:], mode='nearest'))

        # Bottom-up path in PANet
        n2 = p2
        n3 = F.interpolate(self.bu_p2(n2), size=p3.shape[2:], mode='bilinear', align_corners=False) + p3
        n4 = F.interpolate(self.bu_p3(n3), size=p4.shape[2:], mode='bilinear', align_corners=False) + p4
        n5 = F.interpolate(self.bu_p4(n4), size=p5.shape[2:], mode='bilinear', align_corners=False) + p5


        # Combine features
        out = torch.cat([
            F.interpolate(n2, size=(h, w), mode='bilinear', align_corners=False),
            F.interpolate(n3, size=(h, w), mode='bilinear', align_corners=False),
            F.interpolate(n4, size=(h, w), mode='bilinear', align_corners=False),
            F.interpolate(n5, size=(h, w), mode='bilinear', align_corners=False)
        ], dim=1)

        # Final convolution to get the desired number of output channels
        out = self.final_conv(out)

        return out