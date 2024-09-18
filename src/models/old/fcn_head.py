# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead


class SimpleFCNHead(nn.Module):

    def __init__(self, in_channels, channels, num_classes, num_convs=2, kernel_size=3, concat_input=True, dropout_ratio=0.1):
        super(SimpleFCNHead, self).__init__()
        self.in_channels = in_channels
        self.channels = channels
        self.num_classes = num_classes
        self.concat_input = concat_input
        self.convs = nn.ModuleList()
        self.dropout_ratio = 0.1

        for i in range(num_convs):
            padding = kernel_size // 2
            conv = nn.Conv2d(
                in_channels=in_channels if i == 0 else channels,
                out_channels=channels,
                kernel_size=kernel_size,
                padding=padding
            )
            self.convs.append(conv)
            self.convs.append(nn.BatchNorm2d(channels))
            self.convs.append(nn.Dropout2d(self.dropout_ratio))
            self.convs.append(nn.ReLU(inplace=True))

        # Optional: Concatenate input with the output of convs before the final classification layer
        if self.concat_input:
            self.concat_conv = nn.Conv2d(in_channels + channels, channels, kernel_size=1)

        # Final classification layer
        self.final_conv = nn.Conv2d(channels, num_classes, kernel_size=1)

    def forward(self, x):
        original = x
        for layer in self.convs:
            x = layer(x)

        if self.concat_input:
            x = torch.cat([original, x], dim=1)
            x = self.concat_conv(x)

        x = self.final_conv(x)
        return x





class FCNHead(BaseDecodeHead):
    """Fully Convolution Networks for Semantic Segmentation.

    This head is implemented of `FCNNet <https://arxiv.org/abs/1411.4038>`_.

    Args:
        num_convs (int): Number of convs in the head. Default: 2.
        kernel_size (int): The kernel size for convs in the head. Default: 3.
        concat_input (bool): Whether concat the input and output of convs
            before classification layer.
        dilation (int): The dilation rate for convs in the head. Default: 1.
    """

    def __init__(self,
                 num_convs=2,
                 kernel_size=3,
                 concat_input=True,
                 dilation=1,
                 **kwargs):
        assert num_convs >= 0 and dilation > 0 and isinstance(dilation, int)
        self.num_convs = num_convs
        self.concat_input = concat_input
        self.kernel_size = kernel_size
        super(FCNHead, self).__init__(**kwargs)
        if num_convs == 0:
            assert self.in_channels == self.channels

        conv_padding = (kernel_size // 2) * dilation
        convs = []
        for i in range(num_convs):
            _in_channels = self.in_channels if i == 0 else self.channels
            convs.append(
                ConvModule(
                    _in_channels,
                    self.channels,
                    kernel_size=kernel_size,
                    padding=conv_padding,
                    dilation=dilation,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))

        if len(convs) == 0:
            self.convs = nn.Identity()
        else:
            self.convs = nn.Sequential(*convs)
        if self.concat_input:
            self.conv_cat = ConvModule(
                self.in_channels + self.channels,
                self.channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)

    def _forward_feature(self, inputs):
        """Forward function for feature maps before classifying each pixel with
        ``self.cls_seg`` fc.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            feats (Tensor): A tensor of shape (batch_size, self.channels,
                H, W) which is feature map for last layer of decoder head.
        """
        x = self._transform_inputs(inputs)
        print("x.shape: ", x.shape)
        feats = self.convs(x)
        if self.concat_input:
            feats = self.conv_cat(torch.cat([x, feats], dim=1))
        return feats

    def forward(self, inputs):
        """Forward function that computes the segmentation map from inputs."""

        print("inputs: ", inputs)
        x = self._transform_inputs(inputs)  # Aggregate and transform inputs

        print("x.shape: ", x.shape)
        feats = self.convs(x)  # Pass through convolution layers

        print("feats.shape: ", feats.shape)
        
        if self.concat_input:
            feats = self.conv_cat(torch.cat([x, feats], dim=1))  # Concatenate input features with features from convs if required
        
        output = self.cls_seg(feats)  # Final classification layer to predict per-pixel classes
        return output
