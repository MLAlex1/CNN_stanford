"""
This module contains classes and functions that are common across both, one-stage
and two-stage detector implementations. You have to implement some parts here -
walk through the notebooks and you will find instructions on *when* to implement
*what* in this module.
"""
from typing import Dict, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models
from torchvision.models import feature_extraction

import numpy as np


def hello_common():
    print("Hello from common.py!")


class DetectorBackboneWithFPN(nn.Module):
    r"""
    n backbone network: A tiny RegNet model coupled with a Feature
    Pyramid Network (FPN). This model takes in batches of input images with
    shape `(B, 3, H, W)` and gives features from three different FPN levels
    with shapes and total strides upto that level:

        - level p3: (out_channels, H /  8, W /  8)      stride =  8
        - level p4: (out_channels, H / 16, W / 16)      stride = 16
        - level p5: (out_channels, H / 32, W / 32)      stride = 32

    NOTE: We could use any convolutional network architecture that progressively
    downsamples the input image and couple it with FPN. We use a small enough
    backbone that can work with Colab GPU and get decent enough performance.
    """

    def __init__(self, out_channels: int, first_method=True):
        super().__init__()
        self.out_channels = out_channels

        # Initialize with ImageNet pre-trained weights.
        _cnn = models.regnet_x_400mf(pretrained=True)
        self.first_method = first_method

        # Torchvision models only return features from the last level. Detector
        # backbones (with FPN) require intermediate features of different scales.
        # So we wrap the ConvNet with torchvision's feature extractor. Here we
        # will get output features with names (c3, c4, c5) with same stride as
        # (p3, p4, p5) described above.
        self.backbone = feature_extraction.create_feature_extractor(
            _cnn,
            return_nodes={
                "trunk_output.block2": "c3",
                "trunk_output.block3": "c4",
                "trunk_output.block4": "c5",
            },
        )

        # Pass a dummy batch of input images to infer shapes of (c3, c4, c5).
        # Features are a dictionary with keys as defined above. Values are
        # batches of tensors in NCHW format, that give intermediate features
        # from the backbone network.
        dummy_out = self.backbone(torch.randn(2, 3, 224, 224))
        dummy_out_shapes = [(key, value.shape) for key, value in dummy_out.items()]

        print("For dummy input images with shape: (2, 3, 224, 224)")
        for level_name, feature_shape in dummy_out_shapes:
            print(f"Shape of {level_name} features: {feature_shape}")

        ######################################################################
        # TODO: Initialize additional Conv layers for FPN.                   #
        #                                                                    #
        # Create THREE "lateral" 1x1 conv layers to transform (c3, c4, c5)   #
        # such that they all end up with the same `out_channels`.            #
        # Then create THREE "output" 3x3 conv layers to transform the merged #
        # FPN features to output (p3, p4, p5) features.                      #
        # All conv layers must have stride=1 and padding such that features  #
        # do not get downsampled due to 3x3 convs.                           #
        #                                                                    #
        # HINT: You have to use `dummy_out_shapes` defined above to decide   #
        # the input/output channels of these layers.                         #
        ######################################################################
        # This behaves like a Python dict, but makes PyTorch understand that
        # there are trainable weights inside it.
        # Add THREE lateral 1x1 conv and THREE output 3x3 conv layers.
        
        self.fpn_params = nn.ModuleDict()
        if self.first_method:
            s = 1
            f = 1
            for ii, (level_name, feature_shape) in enumerate(dummy_out_shapes):
                self.fpn_params[level_name] = nn.Conv2d(feature_shape[1], out_channels, kernel_size=f) #stride=s)

            f = 3
            for p_name, (level_name, feature_shape) in zip(["p3", "p4", "p5"], dummy_out_shapes):
                w, h = feature_shape[-2:]

                # here it's easy as s=1, thus it simplifies to (f-1)/2
                pw = int(np.ceil(((s-1)*w-s+f)/2))
                ph = int(np.ceil(((s-1)*h-s+f)/2))
                self.fpn_params[p_name] = nn.Conv2d(out_channels, out_channels, kernel_size=f, padding=(pw, ph), stride=s)
        else:
            self.fpn_params['conv5'] = nn.Conv2d(dummy_out['c5'].shape[1], self.out_channels, 1)
            self.fpn_params['conv4'] = nn.Conv2d(dummy_out['c4'].shape[1], self.out_channels, 1)
            self.fpn_params['conv3'] = nn.Conv2d(dummy_out['c3'].shape[1], self.out_channels, 1)

            self.fpn_params['conv_out5'] = nn.Conv2d(self.out_channels, self.out_channels, 3, stride=1, padding=1)
            self.fpn_params['conv_out4'] = nn.Conv2d(self.out_channels, self.out_channels, 3, stride=1, padding=1)
            self.fpn_params['conv_out3'] = nn.Conv2d(self.out_channels, self.out_channels, 3, stride=1, padding=1)

    @property
    def fpn_strides(self):
        """
        Total stride up to the FPN level. For a fixed ConvNet, these values
        are invariant to input image size. You may access these values freely
        to implement your logic in FCOS / Faster R-CNN.
        """
        return {"p3": 8, "p4": 16, "p5": 32}

    def forward(self, images: torch.Tensor):

        # Multi-scale features, dictionary with keys: {"c3", "c4", "c5"}.
        backbone_feats = self.backbone(images)

        fpn_feats = {"p3": None, "p4": None, "p5": None}
        ######################################################################
        # TODO: Fill output FPN features (p3, p4, p5) using RegNet features  #
        # (c3, c4, c5) and FPN conv layers created above.                    #
        # HINT: Use `F.interpolate` to upsample FPN features.                #
        ######################################################################
        if self.first_method:
            prev_c = None
            for p, c, in zip(list(fpn_feats.keys())[::-1], list(backbone_feats.keys())[::-1]): 
                if prev_c == None:
                    prev_c = self.fpn_params[c](backbone_feats[c])
                    fpn_feats[p] = self.fpn_params[p](prev_c)
                else:
                    p_val = F.interpolate(prev_c, scale_factor=2)
                    prev_c = torch.add(p_val, self.fpn_params[c](backbone_feats[c]))
                    fpn_feats[p] = self.fpn_params[p](prev_c)
                
        else:
            c5 = backbone_feats["c5"]; c4 = backbone_feats["c4"]; c3 = backbone_feats["c3"]
            # https://jonathan-hui.medium.com/understanding-feature-pyramid-networks-for-object-detection-fpn-45b227b9106c
            # Remember that c3 has higher resolution than c5 (higher resolution - we can more easily recognise smaller objects,
            # however it's slower because bigger image thus they are not used for object prediction because of the speed)
            # but c5 has a higher semantic value, even though low resolution but because it's quicker to predict on them they are used 
            # for object detection. 
            # F.interpolate is used for upsampling - this is to reconstruct higher resolution layers (downsampled layers have low resolution)
            # from a semantic rich layer (deeper layers). The reconstructed layers are semantic strong but the locations of objects are not 
            # precise after all the downsampling and upsampling. We add lateral connections between reconstructed layers and the corresponding 
            # feature maps to help the detector to predict the location betters. It also acts as skip connections to make training easier
            
            # We apply a 1×1 convolution filter to reduce C5 channel depth to 256-d (self.out_channels) to create an intermediate layer (out5). This becomes the first feature map 
            # layer used for object prediction. As we go down the top-down path, we upsample the previous layer by 2 using nearest neighbors upsampling (F.interpolate(out5, scale_factor=2)).
            # We again apply a 1×1 convolution to the corresponding feature maps in the bottom-up pathway (self.fpn_params['conv4'](c4)). 
            # Then we add them element-wise (F.interpolate(out5, scale_factor=2) + self.fpn_params['conv4'](c4)). 
            # We apply a 3×3 convolution to all merged layers (self.fpn_params['conv_out5'](out5), self.fpn_params['conv_out4'](out4), self.fpn_params['conv_out3'](out3)). 
            # This filter reduces the aliasing effect when merged with the upsampled layer. Because these layers have padding=1, after 3x3 convolutional layer they preserve
            # dimensions so no downsampling happens. 
            out5 = self.fpn_params['conv5'](c5)
                   # reconstructed layer + corresponding feature maps
            out4 = F.interpolate(out5, scale_factor=2) + self.fpn_params['conv4'](c4)
            out3 = F.interpolate(out4, scale_factor=2) + self.fpn_params['conv3'](c3)
            
            fpn_feats["p5"] = self.fpn_params['conv_out5'](out5)
            fpn_feats["p4"] = self.fpn_params['conv_out4'](out4)
            fpn_feats["p3"] = self.fpn_params['conv_out3'](out3)
        
        return fpn_feats


def get_fpn_location_coords(
    shape_per_fpn_level: Dict[str, Tuple],
    strides_per_fpn_level: Dict[str, int],
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
) -> Dict[str, torch.Tensor]:
    """
    Map every location in FPN feature map to a point on the image. This point
    represents the center of the receptive field of this location. We need to
    do this for having a uniform co-ordinate representation of all the locations
    across FPN levels, and GT boxes.

    Args:
        shape_per_fpn_level: Shape of the FPN feature level, dictionary of keys
            {"p3", "p4", "p5"} and feature shapes `(B, C, H, W)` as values.
        strides_per_fpn_level: Dictionary of same keys as above, each with an
            integer value giving the stride of corresponding FPN level.
            See `backbone.py` for more details.

    Returns:
        Dict[str, torch.Tensor]
            Dictionary with same keys as `shape_per_fpn_level` and values as
            tensors of shape `(H * W, 2)` giving `(xc, yc)` co-ordinates of the
            centers of receptive fields of the FPN locations, on input image.
    """

    # Set these to `(N, 2)` Tensors giving absolute location co-ordinates.
    location_coords = {
        level_name: None for level_name, _ in shape_per_fpn_level.items()
    }
    
    # RESULTS DIFFERENt BETWEEN MINE AND 
    # https://github.com/strickland0702/EECS598-008-DLCV/blob/f095a31626f720fcb02503442dfc97eddc54025f/A4/one_stage_detector.ipynb
    # as rows and cols seem inverted. Need more testing
    
    # same but in for loop fashion
    # temp = torch.zeros(H, W, 2)
    # level_stride = 8
    # for i in range(temp.shape[0]):
    #     for j in range(temp.shape[1]):
    #         temp[i, j, 0] = level_stride * (i+0.5)
    #         temp[i, j, 1] = level_stride * (j+0.5)

    for level_name, feat_shape in shape_per_fpn_level.items():
        level_stride = strides_per_fpn_level[level_name]
        H = feat_shape[-2]
        W = feat_shape[-1]
        
        # first arrange rows and columns
        rows = level_stride * (torch.arange(H) + 0.5)
        cols = level_stride * (torch.arange(W) + 0.5)

        # repeat rows W times and cols H times so that we can match two vectors.
        # then join them on the last dim.
        # Basically, we repeat rows W times to create W repetition of each number in rows.
        # This is same as the loop as the index "i" stays same for W times. Then for
        # cols we simply need to repeat a sequence of numbers (1 to W) H times
        rows = rows.repeat(W, 1).t().contiguous().view(-1, 1)
        cols = cols.repeat(H, 1).contiguous().view(-1, 1)

        final = torch.stack([rows, cols], -1).squeeze() # rows and cols inverted as in the other implementation above
        # final = torch.stack([cols, rows], -1).squeeze() # same as temp
        location_coords[level_name] = final
    
#     for level_name, feat_shape in shape_per_fpn_level.items():
#         level_stride = strides_per_fpn_level[level_name]

#         # location_coords[level_name] = temp.flatten(end_dim=1)
#         rows = level_stride * (torch.arange(feat_shape[2], dtype=dtype, device=device) + 0.5)
#         rows = rows.expand(feat_shape[2], feat_shape[3]).t()
#         cols = level_stride * (torch.arange(feat_shape[3], dtype=dtype, device=device) + 0.5)
#         cols = cols.expand(feat_shape[2], feat_shape[3])
#         location_coords[level_name] = torch.stack((cols, rows), dim=2).flatten(end_dim=1)
        
    return location_coords


def IoU(proposals, bboxes):
    """
    Compute intersection over union between sets of bounding boxes.

    Inputs:
    - proposals: Proposals of shape (B, A, H', W', 4)
    - bboxes: Ground-truth boxes from the DataLoader of shape (B, N, 5).
    Each ground-truth box is represented as tuple (x_lr, y_lr, x_rb, y_rb, class).
    If image i has fewer than N boxes, then bboxes[i] will be padded with extra
    rows of -1.

    Outputs:
    - iou_mat: IoU matrix of shape (B, A*H'*W', N) where iou_mat[b, i, n] gives
    the IoU between one element of proposals[b] and bboxes[b, n].

    For this implementation you DO NOT need to filter invalid proposals or boxes;
    in particular you don't need any special handling for bboxxes that are padded
    with -1.
    """
    (_, N, _) = bboxes.size()
    (B, A, H, W, _) = proposals.size()
    # iou_mat = torch.zeros(B, A*H*W, N)
    iou_mat = torch.zeros((B,A*H*W,N),device=proposals.device)*-1
    ##############################################################################
    # TODO: Compute the Intersection over Union (IoU) on proposals and GT boxes. #
    # No need to filter invalid proposals/bboxes (i.e., allow region area <= 0). #
    # You need to ensure your implementation is efficient (no for loops).        #
    # HINT:                                                                      #
    # IoU = Area of Intersection / Area of Union, where
    # Area of Union = Area of Proposal + Area of BBox - Area of Intersection     #
    # and the Area of Intersection can be computed using the top-left corner and #
    # bottom-right corner of proposal and bbox. Think about their relationships. #
    ##############################################################################
    # Replace "pass" statement with your code

    proposals = proposals.reshape(B,A*H*W,4)
    proposals = proposals.unsqueeze(2).repeat(1,1,N,1)

    max_x1 = torch.max(proposals[:,:,:,0],bboxes[:,:,0].unsqueeze(1))
    max_y1 = torch.max(proposals[:,:,:,1],bboxes[:,:,1].unsqueeze(1))
    min_x2 = torch.min(proposals[:,:,:,2],bboxes[:,:,2].unsqueeze(1))
    min_y2 = torch.min(proposals[:,:,:,3],bboxes[:,:,3].unsqueeze(1))
    intersection_area = torch.clamp(min_x2-max_x1,min=0) * torch.clamp(min_y2-max_y1,min=0)

    Area1=(bboxes[:,:,2]-bboxes[:,:,0])*(bboxes[:,:,3]-bboxes[:,:,1])
    Area2=(proposals[:,:,:,2]-proposals[:,:,:,0])*(proposals[:,:,:,3]-proposals[:,:,:,1])

    iou_mat = intersection_area/(Area1.unsqueeze(1)+Area2-intersection_area)
    return iou_mat

def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float = 0.5):
    """
    Non-maximum suppression removes overlapping bounding boxes.

    Args:
        boxes: Tensor of shape (N, 4) giving top-left and bottom-right coordinates
            of the bounding boxes to perform NMS on.
        scores: Tensor of shpe (N, ) giving scores for each of the boxes.
        iou_threshold: Discard all overlapping boxes with IoU > iou_threshold

    Returns:
        keep: torch.long tensor with the indices of the elements that have been
            kept by NMS, sorted in decreasing order of scores;
            of shape [num_kept_boxes]
    """

    if (not boxes.numel()) or (not scores.numel()):
        return torch.zeros(0, dtype=torch.long)

    keep = None
    #############################################################################
    # TODO: Implement non-maximum suppression which iterates the following:     #
    #       1. Select the highest-scoring box among the remaining ones,         #
    #          which has not been chosen in this step before                    #
    #       2. Eliminate boxes with IoU > threshold                             #
    #       3. If any boxes remain, GOTO 1                                      #
    #       Your implementation should not depend on a specific device type;    #
    #       you can use the device of the input if necessary.                   #
    # HINT: You can refer to the torchvision library code:                      #
    # github.com/pytorch/vision/blob/main/torchvision/csrc/ops/cpu/nms_kernel.cpp
    #############################################################################
    
    _, idx_sort = scores.sort(descending=True)
    box_sort = boxes[idx_sort]
    idxs = []
    
    while len(box_sort) != 0:
        highest_box = box_sort[0]
        idx = idx_sort[0]
        idxs.append(idx)
        
        box_sort = box_sort[1:, :]
        idx_sort = idx_sort[1:]
        to_remove = []
        
        # compute IoU between highest scoring box and other boxes
        # we need (B, A, H', W', 4) and  (B, N, 5) shapes for IoU function
        iou_scores = IoU(box_sort.reshape(1, 1, 1, -1, 4), highest_box.reshape(1, 1, -1)).flatten().numpy().astype(float)        
        # remove indices where highest box overlaps with other remaining boxes 
        # with threshold > iou_threshold
        to_remove_idxs = ~(iou_scores > iou_threshold)
        box_sort = box_sort[to_remove_idxs]
        idx_sort = idx_sort[to_remove_idxs]
        
    keep = torch.Tensor(idxs).long()
        
    return keep


def class_spec_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    class_ids: torch.Tensor,
    iou_threshold: float = 0.5,
):
    """
    Wrap `nms` to make it class-specific. Pass class IDs as `class_ids`.
    STUDENT: This depends on your `nms` implementation.

    Returns:
        keep: torch.long tensor with the indices of the elements that have been
            kept by NMS, sorted in decreasing order of scores;
            of shape [num_kept_boxes]
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)
    max_coordinate = boxes.max()
    offsets = class_ids.to(boxes) * (max_coordinate + torch.tensor(1).to(boxes))
    boxes_for_nms = boxes + offsets[:, None]
    keep = nms(boxes_for_nms, scores, iou_threshold)
    return keep
