"""Feature extraction backbone with intermediate layer hooks.

Uses torchvision Wide ResNet-50-2 with hooks on layer2 and layer3,
matching the PatchCore paper (Roth et al., CVPR 2022).
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn
from torch.utils.hooks import RemovableHandle
from torchvision import models


class FeatureExtractor(nn.Module):
    """Extract intermediate features from a pretrained backbone.

    Registers forward hooks on specified layers and collects their outputs.

    Args:
        backbone_name: Name of the torchvision model (default: wide_resnet50_2).
        layers: List of layer names to hook (default: ['layer2', 'layer3']).
        pretrained: Whether to use ImageNet pretrained weights.
    """

    def __init__(
        self,
        backbone_name: str = "wide_resnet50_2",
        layers: list[str] | None = None,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.layers = layers or ["layer2", "layer3"]
        self._features: dict[str, torch.Tensor] = {}
        self._hooks: list[RemovableHandle] = []

        # Load backbone
        if backbone_name == "wide_resnet50_2":
            weights = models.Wide_ResNet50_2_Weights.DEFAULT if pretrained else None
            self.backbone = models.wide_resnet50_2(weights=weights)
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        # Set to eval mode and disable gradients
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad_(False)  # noqa: B006

        # Register hooks
        for layer_name in self.layers:
            layer = dict(self.backbone.named_modules())[layer_name]
            hook = layer.register_forward_hook(self._make_hook(layer_name))
            self._hooks.append(hook)

    def _make_hook(self, name: str) -> Callable:
        """Create a hook function that stores the layer output."""

        def hook_fn(
            module: nn.Module, input: tuple[torch.Tensor, ...], output: torch.Tensor
        ) -> None:
            self._features[name] = output

        return hook_fn

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Extract features from input images.

        Args:
            x: Input tensor of shape (B, 3, H, W).

        Returns:
            Dictionary mapping layer names to their feature tensors.
        """
        self._features.clear()
        # Run forward pass (we don't need the final output)
        with torch.no_grad():
            self.backbone(x)
        return dict(self._features)

    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def __del__(self) -> None:
        self.remove_hooks()
