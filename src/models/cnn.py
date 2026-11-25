"""
Module: cnn.py
Purpose: Adapted Reproduction of the Gravity Spy Convolutional Neural Network (Glanzer et al., 2023).
Author: L4 Group 23 (Ismail Nooruddin, Munir Tariq, Ramsha Arshad)

Architectural Description (Adapted Reproduction):
Follows the 4-block convolutional design of the Gravity Spy CNN (Zevin et al. 2017; Glanzer et al. 2023),
adapted for the public Zenodo v1d1 benchmark (14 shared classes, 140 x 170 float32 spectrograms):
- 4 Convolutional Blocks: Conv2D -> BatchNorm2D -> ReLU -> MaxPool2D(2, 2)
  - Spatial progression: (140, 170) -> (70, 85) -> (35, 42) -> (17, 21) -> (8, 10)
- Classification Head: AdaptiveAvgPool2d((4, 5)) -> Flatten -> Dropout -> Linear(256) -> ReLU -> Dropout -> Linear(num_classes)
- Supports both Single-View (`in_channels=1`, 1.0s spectrogram) and Multi-View (`in_channels=4`, all 4 durations).
- Includes helper `compute_class_weights()` for smoothed inverse-frequency weighted Cross-Entropy loss.
"""

import numpy as np
import torch
import torch.nn as nn


class AdaptedGravitySpyCNN(nn.Module):
    """
    Adapted PyTorch reproduction of the 4-layer Gravity Spy CNN.

    Parameters:
    -----------
    num_classes : int, default=14
        Number of target glitch classes (14 shared classes).
    in_channels : int, default=1
        Number of input spectrogram channels (1 for single view, 4 for multi-view).
    dropout_rate : float, default=0.4
        Dropout probability in the dense classification head for regularization.
    """
    def __init__(self, num_classes=14, in_channels=1, dropout_rate=0.4):
        super(AdaptedGravitySpyCNN, self).__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels

        # Block 1: Large 5x5 initial receptive field captures broad burst/line morphology
        # Input: (B, in_channels, 140, 170) -> Output: (B, 16, 70, 85)
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Block 2: Captures mid-level spectral-temporal contours (teardrops, fins, arches)
        # Input: (B, 16, 70, 85) -> Output: (B, 32, 35, 42)
        self.block2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Block 3: Captures higher-order shape patterns (repeating pulses, harmonic bands)
        # Input: (B, 32, 35, 42) -> Output: (B, 64, 17, 21)
        self.block3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Block 4: High-level semantic glitch features
        # Input: (B, 64, 17, 21) -> Output: (B, 64, 8, 10)
        self.block4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Spatial pooling keeps feature vector compact (64 * 4 * 5 = 1,280 dims) to prevent overfitting
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 5))

        # Fully Connected Classification Head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(64 * 4 * 5, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate * 0.75),
            nn.Linear(256, num_classes)
        )

    def extract_features(self, x):
        """Returns the 256-dim penultimate embedding (useful for future multimodal fusion)."""
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.adaptive_pool(x)
        x = self.classifier[0](x)  # Flatten
        x = self.classifier[1](x)  # Dropout
        x = self.classifier[2](x)  # Linear -> 256
        x = self.classifier[3](x)  # ReLU
        return x

    def forward(self, x):
        """Forward pass emitting raw class logits of shape (batch_size, num_classes)."""
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.adaptive_pool(x)
        logits = self.classifier(x)
        return logits


def compute_class_weights(labels_idx, num_classes=14, power=0.5):
    """
    Computes smoothed inverse-frequency class weights for CrossEntropyLoss.

    Using `power=0.5` (square-root dampening) boosts rare classes (e.g., Chirp, Tomte)
    without causing unstable gradients or excessive false positives.

    Parameters:
    -----------
    labels_idx : array-like of int
        Integer training labels in [0, num_classes - 1].
    num_classes : int
        Total number of classes.
    power : float, default=0.5
        Exponent applied to inverse class frequency (1.0 = full inverse, 0.5 = smoothed).

    Returns:
    --------
    weights_tensor : torch.FloatTensor of shape (num_classes,)
        Normalized weights with mean 1.0.
    """
    counts = np.bincount(np.asarray(labels_idx, dtype=np.int64), minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    raw_weights = (np.sum(counts) / (num_classes * counts)) ** power
    norm_weights = raw_weights / np.mean(raw_weights)
    return torch.tensor(norm_weights, dtype=torch.float32)


def count_parameters(model):
    """Returns total number of trainable parameters in the PyTorch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Self-test single-view and multi-view forward/backward passes
    for ch in [1, 4]:
        net = AdaptedGravitySpyCNN(num_classes=14, in_channels=ch)
        dummy_x = torch.randn(8, ch, 140, 170)
        logits = net(dummy_x)
        loss = logits.sum()
        loss.backward()
        print(f"[OK] AdaptedGravitySpyCNN(in_channels={ch}) | Input: {tuple(dummy_x.shape)} -> Logits: {tuple(logits.shape)} | Params: {count_parameters(net):,}")
