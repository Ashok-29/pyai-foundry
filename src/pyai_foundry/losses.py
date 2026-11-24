"""Unified Loss Registry covering Classification, Regression, Vision, Contrastive & LLM tasks."""

from __future__ import annotations

from typing import Any, Callable, Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F

_LOSS_REGISTRY: Dict[str, Callable[..., nn.Module]] = {}


def register_loss(name: str) -> Callable[[Callable[..., nn.Module]], Callable[..., nn.Module]]:
    key = name.strip().lower()

    def decorator(fn: Callable[..., nn.Module]) -> Callable[..., nn.Module]:
        _LOSS_REGISTRY[key] = fn
        return fn

    return decorator


def list_losses() -> List[str]:
    return sorted(_LOSS_REGISTRY.keys())


def build_loss(name: str, **kwargs: Any) -> nn.Module:
    key = name.strip().lower()
    if key not in _LOSS_REGISTRY:
        raise KeyError(f"Unknown loss '{name}'. Available: {', '.join(list_losses())}")
    return _LOSS_REGISTRY[key](**kwargs)


@register_loss("focal")
class FocalLoss(nn.Module):
    """Multi-class Focal Loss for imbalanced classification."""

    def __init__(self, gamma: float = 2.0, alpha: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        return (self.alpha * ((1.0 - pt) ** self.gamma) * ce).mean()


@register_loss("label_smoothing_ce")
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing: float = 0.1) -> None:
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, targets, label_smoothing=self.smoothing)


@register_loss("charbonnier")
class CharbonnierLoss(nn.Module):
    def __init__(self, eps: float = 1e-3) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.sqrt((pred - target) ** 2 + self.eps**2))


@register_loss("dice")
class SoftDiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        inter = (probs * targets).sum()
        denom = probs.sum() + targets.sum()
        return 1.0 - (2.0 * inter + self.smooth) / (denom + self.smooth)


@register_loss("infonce")
class InfoNCELoss(nn.Module):
    """CLIP / SimCLR-style symmetric contrastive loss over normalized embedding pairs."""

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        logits = (z1 @ z2.T) / self.temperature
        labels = torch.arange(z1.shape[0], device=z1.device)
        return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


@register_loss("causal_lm")
class CausalLMLoss(nn.Module):
    """Autoregressive next-token cross-entropy loss for decoder-only LLMs."""

    def __init__(self, ignore_index: int = -100) -> None:
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=self.ignore_index,
        )


@register_loss("huber")
class HuberRegressionLoss(nn.Module):
    def __init__(self, delta: float = 1.0) -> None:
        super().__init__()
        self.delta = delta

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.huber_loss(pred, target, delta=self.delta)
