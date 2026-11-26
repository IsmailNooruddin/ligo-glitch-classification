"""
Module: transfer_models.py
Purpose: Step 15 (Novelty 1) — Modern Pretrained Vision Backbones for Transfer Learning on LIGO Glitches.
Author: L4 Group 23 (Ismail Nooruddin, Munir Tariq, Ramsha Arshad)

Supports three modern ImageNet-pretrained vision paradigms:
1. ResNet-18      (`resnet18`)      — Residual CNN backbone (~11.2M params, 512-dim embedding)
2. ConvNeXt-Tiny  (`convnext_tiny`) — Modernized pure-ConvNet architecture (~27.8M params, 768-dim embedding)
3. ViT-B/16       (`vit_b_16`)      — Vision Transformer with global self-attention (~85.8M params, 768-dim embedding)

Key Design Features:
- Accepts either Single-View (1, 140, 170) -> auto-repeated to 3 channels, OR
  Multi-View RGB Pseudo-Spectrograms (3, 140, 170) mapping [0.5s, 1.0s, 2.0s] -> [R, G, B].
- Resizes to (224, 224) and applies ImageNet channel normalization on-device inside forward(),
  ensuring GPU acceleration and exact compatibility with ViT-B/16's fixed 14x14 patch grid.
- Supports partial freezing (`set_backbone_trainable`) and differential learning rates (`get_param_groups`).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


SUPPORTED_BACKBONES = ("resnet18", "convnext_tiny", "vit_b_16")


class TransferLIGOClassifier(nn.Module):
    """
    Unified Transfer Learning wrapper around torchvision ImageNet backbones for LIGO glitch classification.

    Parameters:
    -----------
    backbone_name : str
        One of 'resnet18', 'convnext_tiny', or 'vit_b_16'.
    num_classes : int, default=14
        Number of target LIGO glitch classes.
    pretrained : bool, default=True
        Loads ImageNet-1K pretrained weights (`weights="DEFAULT"`).
    dropout_rate : float, default=0.3
        Dropout probability before the final linear classification head.
    input_size : tuple of int, default=(224, 224)
        Target spatial resolution (224x224 is strictly required for vit_b_16).
    """
    def __init__(self, backbone_name="resnet18", num_classes=14, pretrained=True,
                 dropout_rate=0.3, input_size=(224, 224)):
        super().__init__()
        if backbone_name not in SUPPORTED_BACKBONES:
            raise ValueError(f"Unsupported backbone '{backbone_name}'. Choose from {SUPPORTED_BACKBONES}.")

        self.backbone_name = backbone_name
        self.num_classes = num_classes
        self.input_size = input_size

        # Register ImageNet normalization constants as buffers so .to(device) moves them automatically
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("img_mean", mean)
        self.register_buffer("img_std", std)

        weights_arg = "DEFAULT" if pretrained else None

        if backbone_name == "resnet18":
            net = models.resnet18(weights=weights_arg)
            in_features = net.fc.in_features
            net.fc = nn.Identity()
            self.backbone = net

        elif backbone_name == "convnext_tiny":
            net = models.convnext_tiny(weights=weights_arg)
            in_features = net.classifier[2].in_features
            # Preserve LayerNorm2d + Flatten in classifier[:2], replace final Linear with Identity
            net.classifier[2] = nn.Identity()
            self.backbone = net

        elif backbone_name == "vit_b_16":
            net = models.vit_b_16(weights=weights_arg)
            in_features = net.heads.head.in_features
            net.heads = nn.Identity()
            self.backbone = net

        self.embed_dim = in_features
        self.head = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, num_classes)
        )

    def _preprocess(self, x):
        """Ensures 3 channels, resizes to (224, 224), and applies ImageNet normalization."""
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        elif x.shape[1] != 3:
            raise ValueError(f"Expected 1 or 3 input channels, got shape {tuple(x.shape)}.")

        if x.shape[2:] != self.input_size:
            x = F.interpolate(x, size=self.input_size, mode="bilinear", align_corners=False)

        x = (x - self.img_mean) / self.img_std
        return x

    def extract_features(self, x):
        """Returns penultimate semantic visual embedding of shape (B, embed_dim)."""
        x = self._preprocess(x)
        return self.backbone(x)

    def forward(self, x):
        """Forward pass emitting raw class logits of shape (B, num_classes)."""
        feats = self.extract_features(x)
        return self.head(feats)

    def get_param_groups(self, backbone_lr=1e-4, head_lr=1e-3, weight_decay=1e-2):
        """
        Returns differential learning-rate parameter groups so pretrained ImageNet weights
        adapt gently while the newly initialized classification head converges rapidly.
        """
        return [
            {"params": self.backbone.parameters(), "lr": backbone_lr, "weight_decay": weight_decay},
            {"params": self.head.parameters(), "lr": head_lr, "weight_decay": weight_decay},
        ]

    def set_backbone_trainable(self, trainable=True):
        """Freezes or unfreezes backbone parameters (for linear probing vs full fine-tuning)."""
        for p in self.backbone.parameters():
            p.requires_grad = trainable


def count_trainable_parameters(model):
    """Returns total number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    for name in SUPPORTED_BACKBONES:
        net = TransferLIGOClassifier(backbone_name=name, num_classes=14, pretrained=False)
        for ch in [1, 3]:
            x = torch.randn(2, ch, 140, 170)
            logits = net(x)
            assert logits.shape == (2, 14), f"Unexpected output shape {logits.shape} for {name}"
            feats = net.extract_features(x)
            assert feats.shape == (2, net.embed_dim), f"Unexpected embed shape {feats.shape} for {name}"
        loss = logits.sum()
        loss.backward()
        print(f"[OK] {name:<14} | Embed={net.embed_dim:4d} | Params={count_trainable_parameters(net):>10,} | 1ch & 3ch (140x170) -> (2, 14) passed!")
