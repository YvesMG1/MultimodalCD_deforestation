# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

from mmseg.models import builder
from mmseg.models.builder import SEGMENTORS
from mmseg.models.segmentors.base import BaseSegmentor
from mmseg.models.segmentors.encoder_decoder import EncoderDecoder

import torch.nn as nn
import torch.nn.functional as F


class CDTemporalModel(nn.Module):
    """Custom model for change detection using backbone, neck, and head components."""

    def __init__(self, backbone, neck, decode_head, auxiliary_head=None, frozen_backbone=False, mode='Concat'):
        super(CDTemporalModel, self).__init__()
        self.backbone = backbone
        self.neck = neck
        self.decode_head = decode_head
        self.auxiliary_head = auxiliary_head
        self.mode = mode

        if frozen_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, img1, img2):
        # Pass images through the backbone
        feat1 = self.backbone(img1.unsqueeze(2))
        feat2 = self.backbone(img2.unsqueeze(2))

        # Calculate the difference or any other operation
        if self.mode == 'Concat':
            feats = torch.cat((feat1, feat2), 2)
        else:
            feats = torch.abs(feat1 - feat2)

        # Pass the feature through the neck if it exists
        if self.neck is not None:
            feats = self.neck(feats)

        # Decode the features to segmentation map
        output = self.decode_head(feats)

        # If there is an auxiliary head, use it for deep supervision during training
        if self.auxiliary_head is not None:
            auxiliary_output = self.auxiliary_head(feats)
            return output, auxiliary_output

        return output


def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > input_w:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    return F.interpolate(input, size, scale_factor, mode, align_corners)


class TemporalEncoderDecoder(EncoderDecoder):
    """Encoder Decoder segmentors.

    EncoderDecoder typically consists of backbone, neck, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.

    The backbone should return plain embeddings.
    The neck can process these to make them suitable for the chosen heads.
    The heads perform the final processing that will return the output.
    """

    def __init__(self,
                 backbone,
                 decode_head,
                 neck=None,
                 auxiliary_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None,
                 frozen_backbone=False):
        super(EncoderDecoder, self).__init__(init_cfg)
        if pretrained is not None:
            assert backbone.get('pretrained') is None, \
                'both backbone and segmentor set pretrained weight'
            backbone.pretrained = pretrained
        self.backbone = builder.build_backbone(backbone)
        
        if frozen_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        if neck is not None:
            self.neck = builder.build_neck(neck)
        self._init_decode_head(decode_head)
        self._init_auxiliary_head(auxiliary_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        print("Hello world from TemporalEncoderDecoder")
        print(" This is a test for the new class ")
        print("-----------------")
        


        assert self.with_decode_head

    def encode_decode(self, img, img_metas):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        x = self.extract_feat(img)
        out = self._decode_head_forward_test(x, img_metas)
        
        #### size calculated over last two dimensions ###
        size = img.shape[-2:]
        
        out = resize(
            input=out,
            size=size,
            mode='bilinear',
            align_corners=self.align_corners)
        return out
      
    def slide_inference(self, img, img_meta, rescale):
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.
        """

        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        
        #### size and bactch size over last two dimensions ###
        img_size = img.size()
        batch_size = img_size[0]
        h_img = img_size[-2]
        w_img = img_size[-1]
        out_channels = self.out_channels
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = img.new_zeros((batch_size, out_channels, h_img, w_img))
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                
                if len(img_size) == 4:
                    
                    crop_img = img[:, :, y1:y2, x1:x2]
                
                elif len(img_size) == 5:
                    
                    crop_img = img[:, :, :, y1:y2, x1:x2]
                
                
                
                crop_seg_logit = self.encode_decode(crop_img, img_meta)
                preds += F.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        if torch.onnx.is_in_onnx_export():
            # cast count_mat to constant while exporting to ONNX
            count_mat = torch.from_numpy(
                count_mat.cpu().detach().numpy()).to(device=img.device)
        preds = preds / count_mat

        if rescale:
            # remove padding area
            #### size over last two dimensions ###
            resize_shape = img_meta[0]['img_shape'][:2]
            preds = preds[:, :, :resize_shape[0], :resize_shape[1]]
            preds = resize(
                preds,
                size=img_meta[0]['ori_shape'][:2],
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)
        return preds

    def whole_inference(self, img, img_meta, rescale):
        """Inference with full image."""

        seg_logit = self.encode_decode(img, img_meta)
        if rescale:
            # support dynamic shape for onnx
            if torch.onnx.is_in_onnx_export():
                size = img.shape[-2:]
            else:
                # remove padding area
                resize_shape = img_meta[0]['img_shape'][:2] 
                seg_logit = seg_logit[:, :, :resize_shape[0], :resize_shape[1]]
                size = img_meta[0]['ori_shape'][:2]
            seg_logit = resize(
                seg_logit,
                size=size,
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)

        return seg_logit

    def inference(self, img, img_meta, rescale):
        """Inference with slide/whole style.

        Args:
            img (Tensor): The input image of shape (N, 3, H, W).
            img_meta (dict): Image info dict where each dict has: 'img_shape',
                'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            rescale (bool): Whether rescale back to original shape.

        Returns:
            Tensor: The output segmentation map.
        """

        assert self.test_cfg.mode in ['slide', 'whole']
        ori_shape = img_meta[0]['ori_shape']
        assert all(_['ori_shape'] == ori_shape for _ in img_meta)
        if self.test_cfg.mode == 'slide':
            seg_logit = self.slide_inference(img, img_meta, rescale)
        else:
            seg_logit = self.whole_inference(img, img_meta, rescale)
            
        if self.out_channels == 1:
            output = F.sigmoid(seg_logit)
        else:
            output = F.softmax(seg_logit, dim=1)

        flip = (
            img_meta[0]["flip"] if "flip" in img_meta[0] else False
        )  ##### if flip key is not there d not apply it
        if flip:
            flip_direction = img_meta[0]["flip_direction"]
            assert flip_direction in ["horizontal", "vertical"]
            if flip_direction == "horizontal":
                output = output.flip(dims=(3,))
            elif flip_direction == "vertical":
                output = output.flip(dims=(2,))
        return output

    def simple_test(self, img, img_meta, rescale=True):
        """Simple test with single image."""
        seg_logit = self.inference(img, img_meta, rescale)
        if self.out_channels == 1:
            seg_pred = (seg_logit > self.decode_head.threshold).to(seg_logit).squeeze(1)
        else:
            seg_pred = seg_logit.argmax(dim=1)
        if torch.onnx.is_in_onnx_export():

            seg_pred = seg_pred.unsqueeze(0)
            return seg_pred
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred


@SEGMENTORS.register_module()
class CDTemporalEncoderDecoder(EncoderDecoder):
    """Encoder Decoder segmentors.

    EncoderDecoder typically consists of backbone, neck, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.

    The backbone should return plain embeddings.
    The neck can process these to make them suitable for the chosen heads.
    The heads perform the final processing that will return the output.
    """

    def __init__(self,
                 backbone,
                 decode_head,
                 neck=None,
                 auxiliary_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None,
                 frozen_backbone=True):
        super(EncoderDecoder, self).__init__(init_cfg)
        if pretrained is not None:
            assert backbone.get('pretrained') is None, \
                'both backbone and segmentor set pretrained weight'
            backbone.pretrained = pretrained
        self.backbone = builder.build_backbone(backbone)
        
        if frozen_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        if neck is not None:
            self.neck = builder.build_neck(neck)
        self._init_decode_head(decode_head)
        self._init_auxiliary_head(auxiliary_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        print("Hello world from TemporalEncoderDecoder")
        print(" This is a test for the new class ")
        print("-----------------")
        


        assert self.with_decode_head

    def encode_decode(self, img1, img2, img_metas):
        """Process two images, compute difference, and decode into a semantic segmentation map."""
        # Encode both images
        feat1 = self.extract_feat(img1)
        feat2 = self.extract_feat(img2)
        
        # Compute the difference between the features
        diff_feat = torch.abs(feat1 - feat2)
        
        # Decode the feature difference
        #out = self._decode_head_forward_test(diff_feat, img_metas)
        size = img1.shape[-2:]  # Assuming img1 and img2 are the same size
        out = resize(out, size=size, mode='bilinear', align_corners=self.align_corners)
        return out
      
