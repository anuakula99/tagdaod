"""Detectors.

* "dino": Faster R-CNN on a FROZEN DINOv2 ViT (timm). Only a small simple feature
  pyramid, the RPN and the box head are trained. This is the pseudo-labeller.
* "r50":  standard torchvision Faster R-CNN ResNet-50 FPN. This is the student that
  runs at test time and in the demo.
"""
import math
from collections import OrderedDict

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet50_Weights
from torchvision.models.detection import (FasterRCNN, FasterRCNN_ResNet50_FPN_Weights,
                                          fasterrcnn_resnet50_fpn)
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import MultiScaleRoIAlign

DINO_MODELS = {
    "vits": "vit_small_patch14_dinov2.lvd142m",
    "vitb": "vit_base_patch14_dinov2.lvd142m",
    "vitl": "vit_large_patch14_dinov2.lvd142m",
}


def _proj(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 1, bias=False), nn.GroupNorm(32, cout),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.GroupNorm(32, cout))


class DinoV2SimpleFPN(nn.Module):
    """Frozen DINOv2 + ViTDet-style simple feature pyramid.

    The image is resized so that one 14x14 ViT patch covers exactly 16x16 input pixels.
    The single stride-16 token map then gives pyramid levels at strides 8/16/32/64/128,
    which line up exactly with the RoIAlign scales.
    """

    def __init__(self, model_name, pretrained=True, out_channels=256):
        super().__init__()
        self.vit = timm.create_model(model_name, pretrained=pretrained, num_classes=0,
                                     dynamic_img_size=True)
        for p in self.vit.parameters():
            p.requires_grad_(False)
        self.vit.eval()
        dim = self.vit.embed_dim
        self.patch = self.vit.patch_embed.patch_size[0]
        self.n_prefix = getattr(self.vit, "num_prefix_tokens", 1)
        self.up = nn.Sequential(nn.ConvTranspose2d(dim, dim // 2, 2, 2), nn.GroupNorm(32, dim // 2), nn.GELU())
        self.p2 = _proj(dim // 2, out_channels)  # stride 8
        self.p3 = _proj(dim, out_channels)       # stride 16
        self.p4 = _proj(dim, out_channels)       # stride 32
        self.out_channels = out_channels

    def train(self, mode=True):
        super().train(mode)
        self.vit.eval()  # the ViT is frozen: always eval mode
        return self

    def forward(self, x):
        b, _, h, w = x.shape
        gh, gw = math.ceil(h / 16), math.ceil(w / 16)
        xin = F.interpolate(x, size=(gh * self.patch, gw * self.patch), mode="bilinear", align_corners=False)
        with torch.no_grad():
            tokens = self.vit.forward_features(xin)[:, self.n_prefix:, :]
        f = tokens.transpose(1, 2).reshape(b, -1, gh, gw).float()
        out = OrderedDict()
        out["0"] = self.p2(self.up(f))
        out["1"] = self.p3(f)
        out["2"] = self.p4(F.max_pool2d(f, 2, ceil_mode=True))
        out["3"] = F.max_pool2d(out["2"], 2, ceil_mode=True)
        out["pool"] = F.max_pool2d(out["3"], 1, 2)
        return out


def build_model(cfg):
    """cfg keys: arch, num_classes, pretrained ('coco'|'imagenet'|'none'), dino_model,
    min_size, max_size."""
    nc = cfg["num_classes"] + 1
    pre = cfg.get("pretrained", "coco")
    common = dict(min_size=cfg["min_size"], max_size=cfg["max_size"])
    if cfg["arch"] == "r50":
        if pre == "coco":
            m = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1, **common)
        else:
            m = fasterrcnn_resnet50_fpn(
                weights=None, num_classes=91,
                weights_backbone=ResNet50_Weights.IMAGENET1K_V1 if pre == "imagenet" else None, **common)
        m.roi_heads.box_predictor = FastRCNNPredictor(m.roi_heads.box_predictor.cls_score.in_features, nc)
        return m
    if cfg["arch"] == "dino":
        name = DINO_MODELS.get(cfg.get("dino_model", "vitb"), cfg.get("dino_model"))
        backbone = DinoV2SimpleFPN(name, pretrained=pre != "none")
        anchors = AnchorGenerator(sizes=((32,), (64,), (128,), (256,), (512,)),
                                  aspect_ratios=((0.5, 1.0, 2.0),) * 5)
        roi = MultiScaleRoIAlign(featmap_names=["0", "1", "2", "3"], output_size=7, sampling_ratio=2)
        return FasterRCNN(backbone, num_classes=nc, rpn_anchor_generator=anchors, box_roi_pool=roi, **common)
    raise ValueError(f"unknown arch {cfg['arch']}")


def trainable_state_dict(model, cfg):
    sd = model.state_dict()
    if cfg["arch"] == "dino" and cfg.get("pretrained", "coco") != "none":
        # the frozen ViT is reloaded from timm, so do not store ~350 MB of fixed weights
        sd = {k: v for k, v in sd.items() if not k.startswith("backbone.vit.")}
    return sd


def save_checkpoint(model, cfg, path, **extra):
    torch.save({"cfg": cfg, "model": trainable_state_dict(model, cfg), **extra}, path)


def load_checkpoint(path, device="cpu"):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = dict(ck["cfg"])
    build_cfg = dict(cfg)
    if cfg["arch"] == "r50":
        build_cfg["pretrained"] = "none"  # all weights come from the checkpoint
    model = build_model(build_cfg)
    missing, unexpected = model.load_state_dict(ck["model"], strict=False)
    bad = [k for k in missing if not k.startswith("backbone.vit.")] + list(unexpected)
    if bad:
        raise RuntimeError(f"checkpoint mismatch: {bad[:5]}")
    return model.to(device).eval(), cfg
