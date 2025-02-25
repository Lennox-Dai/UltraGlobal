# Copyright (c) OpenMMLab. All rights reserved
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import auto_fp16
 
from ..builder import NECKS
from .fpn import FPN
 
 
@NECKS.register_module()
class PAFPN(FPN):
    """Path Aggregation Network for Instance Segmentation.
    This is an implementation of the `PAFPN in Path Aggregation Network
    <https://arxiv.org/abs/1803.01534>`_.
    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale)
        num_outs (int): Number of output scales.
        start_level (int): Index of the start input backbone level used to
            build the feature pyramid. Default: 0.
        end_level (int): Index of the end input backbone level (exclusive) to
            build the feature pyramid. Default: -1, which means the last level.
        add_extra_convs (bool | str): If bool, it decides whether to add conv
            layers on top of the original feature maps. Default to False.
            If True, it is equivalent to `add_extra_convs='on_input'`.
            If str, it specifies the source feature map of the extra convs.
            Only the following options are allowed
            - 'on_input': Last feat map of neck inputs (i.e. backbone feature).
            - 'on_lateral':  Last feature map after lateral convs.
            - 'on_output': The last output feature map after fpn convs.
        relu_before_extra_convs (bool): Whether to apply relu before the extra
            conv. Default: False.
        no_norm_on_lateral (bool): Whether to apply norm on lateral.
            Default: False.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (str): Config dict for activation layer in ConvModule.
            Default: None.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """
 
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        super(PAFPN, self).__init__(
            in_channels,
            out_channels,
            num_outs,
            start_level,
            end_level,
            add_extra_convs,
            relu_before_extra_convs,
            no_norm_on_lateral,
            conv_cfg,
            norm_cfg,
            act_cfg,
            init_cfg=init_cfg)
        # add extra bottom up pathway
        self.downsample_convs = nn.ModuleList()
        self.pafpn_convs = nn.ModuleList()
        for i in range(self.start_level + 1, self.backbone_end_level):
            d_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                stride=2,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)
            pafpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)
            self.downsample_convs.append(d_conv)
            self.pafpn_convs.append(pafpn_conv)
 
    @auto_fp16()
    def forward(self, inputs):
        # [torch.Size([2, 256, 75, 75])
        #  torch.Size([2, 512, 38, 38])
        #  torch.Size([2, 1024, 19, 19])
        #  torch.Size([2, 2048, 10, 10])]
        """Forward function."""
        assert len(inputs) == len(self.in_channels)
 
        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])  # 0
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        # print(self.lateral_convs)
        # ModuleList(
        #   (0): ConvModule(
        #     (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
        #   )
        #   (1): ConvModule(
        #     (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1))
        #   )
        #   (2): ConvModule(
        #     (conv): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1))
        #   )
        #   (3): ConvModule(
        #     (conv): Conv2d(2048, 256, kernel_size=(1, 1), stride=(1, 1))
        #   )
        # )
        # print(laterals)
        # [torch.Size([2, 256, 75, 75])
        #  torch.Size([2, 256, 38, 38])
        #  torch.Size([2, 256, 19, 19])
        #  torch.Size([2, 256, 10, 10])]
 
        # build top-down path
        used_backbone_levels = len(laterals)  # 4
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            # fix runtime error of "+=" inplace operation in PyTorch 1.10
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=prev_shape, mode='nearest')
 
        # build outputs
        # part 1: from original levels
        inter_outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]
        # print(self.fpn_convs)
        # ModuleList(
        #   (0): ConvModule(
        #     (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #   )
        #   (1): ConvModule(
        #     (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #   )
        #   (2): ConvModule(
        #     (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #   )
        #   (3): ConvModule(
        #     (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #   )
        # )
 
        # print(self.downsample_convs)
        # ModuleList(
        #   (0): ConvModule(
        #     (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        #   )
        #   (1): ConvModule(
        #     (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        #   )
        #   (2): ConvModule(
        #     (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        #   )
        # )
 
        # part 2: add bottom-up path
        for i in range(0, used_backbone_levels - 1):
            inter_outs[i + 1] += self.downsample_convs[i](inter_outs[i])
 
        # print(self.pafpn_convs)
        # ModuleList(
        #   (0): ConvModule(
        #     (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #   )
        #   (1): ConvModule(
        #     (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #   )
        #   (2): ConvModule(
        #     (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #   )
        # )
        outs = []
        outs.append(inter_outs[0])
        outs.extend([
            self.pafpn_convs[i - 1](inter_outs[i])
            for i in range(1, used_backbone_levels)
        ])
 
        # part 3: add extra levels
        if self.num_outs > len(outs):  # 5,4
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:  # False
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.add_extra_convs == 'on_input':
                    orig = inputs[self.backbone_end_level - 1]
                    outs.append(self.fpn_convs[used_backbone_levels](orig))
                elif self.add_extra_convs == 'on_lateral':
                    outs.append(self.fpn_convs[used_backbone_levels](
                        laterals[-1]))
                elif self.add_extra_convs == 'on_output':
                    outs.append(self.fpn_convs[used_backbone_levels](outs[-1]))
                else:
                    raise NotImplementedError
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        # print(outs)
        # [torch.Size([2, 256, 75, 75])
        #  torch.Size([2, 256, 38, 38])
        #  torch.Size([2, 256, 19, 19])
        #  torch.Size([2, 256, 10, 10])]
        return tuple(outs)
    
input_tensor = [
    torch.randn(1, 256, 75, 75),  # Example tensor for the first scale
    torch.randn(1, 512, 38, 38),  # Example tensor for the second scale
    torch.randn(1, 1024, 19, 19), # Example tensor for the third scale
    torch.randn(1, 2048, 10, 10)  # Example tensor for the fourth scale
]

# Instantiate the PAFPN network
pafpn = PAFPN(
    in_channels=[256, 512, 1024, 2048],
    out_channels=256,
    num_outs=5,
    start_level=0,
    end_level=-1,
    add_extra_convs=False,
    relu_before_extra_convs=False,
    no_norm_on_lateral=False
)

# Forward pass through the network
outputs = pafpn(input_tensor)

# Print the shapes of the outputs
for i, output in enumerate(outputs):
    print(f"Output {i + 1}: {output.shape}")