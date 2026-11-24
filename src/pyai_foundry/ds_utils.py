"""Data Science & ML Supporting Functions: Scalers, Splits, Outliers, Stats, & Feature Engineering."""

from __future__ import annotations

import math
from typing import Dict, List, Tuple
import torch


class StandardScaler:
    """Zero-mean, unit-variance feature scaler supporting fit/transform/inverse_transform."""

    def __init__(self, eps: float = 1e-8) -> None:
        self.eps = eps
        self.mean_: torch.Tensor | None = None
        self.std_: torch.Tensor | None = None

    def fit(self, x: torch.Tensor) -> "StandardScaler":
        self.mean_ = x.mean(dim=0, keepdim=True)
        self.std_ = x.std(dim=0, unbiased=False, keepdim=True).clamp(min=self.eps)
        return self

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Call fit() before transform().")
        return (x - self.mean_) / self.std_

    def fit_transform(self, x: torch.Tensor) -> torch.Tensor:
        return self.fit(x).transform(x)

    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Call fit() before inverse_transform().")
        return x * self.std_ + self.mean_


class MinMaxScaler:
    """Scales features to a target [min_val, max_val] range."""

    def __init__(self, feature_range: Tuple[float, float] = (0.0, 1.0), eps: float = 1e-8) -> None:
        self.low, self.high = feature_range
        self.eps = eps
        self.min_: torch.Tensor | None = None
        self.max_: torch.Tensor | None = None

    def fit(self, x: torch.Tensor) -> "MinMaxScaler":
        self.min_ = x.min(dim=0, keepdim=True).values
        self.max_ = x.max(dim=0, keepdim=True).values
        return self

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        if self.min_ is None or self.max_ is None:
            raise RuntimeError("Call fit() before transform().")
        scale = (self.max_ - self.min_).clamp(min=self.eps)
        return self.low + (x - self.min_) / scale * (self.high - self.low)

    def fit_transform(self, x: torch.Tensor) -> torch.Tensor:
        return self.fit(x).transform(x)


def train_val_test_split(
    n_samples: int,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Deterministic index split for train / validation / test partitions."""
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_samples, generator=g)
    n_test = max(1, int(n_samples * test_ratio))
    n_val = max(1, int(n_samples * val_ratio))
    test_idx = perm[:n_test]
    val_idx = perm[n_test : n_test + n_val]
    train_idx = perm[n_test + n_val :]
    return train_idx, val_idx, test_idx


def k_fold_indices(n_samples: int, k: int = 5, seed: int = 42) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Generate K-fold (train_idx, val_idx) splits."""
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_samples, generator=g)
    folds = torch.chunk(perm, k)
    splits: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for i in range(k):
        val_idx = folds[i]
        train_idx = torch.cat([folds[j] for j in range(k) if j != i])
        splits.append((train_idx, val_idx))
    return splits


def detect_outliers_iqr(x: torch.Tensor, factor: float = 1.5) -> torch.Tensor:
    """Return boolean mask of IQR outliers per element."""
    q1 = torch.quantile(x.float(), 0.25, dim=0, keepdim=True)
    q3 = torch.quantile(x.float(), 0.75, dim=0, keepdim=True)
    iqr = q3 - q1
    return (x < (q1 - factor * iqr)) | (x > (q3 + factor * iqr))


def correlation_matrix(x: torch.Tensor) -> torch.Tensor:
    """Compute Pearson correlation matrix (D, D) across feature columns."""
    xc = x.float() - x.float().mean(dim=0, keepdim=True)
    cov = (xc.T @ xc) / max(1, x.shape[0] - 1)
    std = xc.std(dim=0, unbiased=True).clamp(min=1e-8)
    return (cov / torch.outer(std, std)).clamp(-1.0, 1.0)


def bootstrap_confidence_interval(
    values: torch.Tensor, n_bootstrap: int = 500, alpha: float = 0.05, seed: int = 42
) -> Dict[str, float]:
    """Compute non-parametric bootstrap mean and confidence interval."""
    v = values.float().flatten()
    n = v.numel()
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, n, (n_bootstrap, n), generator=g)
    means = v[idx].mean(dim=1)
    low = torch.quantile(means, alpha / 2.0).item()
    high = torch.quantile(means, 1.0 - alpha / 2.0).item()
    return {
        "mean": round(v.mean().item(), 5),
        "std": round(v.std(unbiased=True).item() if n > 1 else 0.0, 5),
        "ci_low": round(low, 5),
        "ci_high": round(high, 5),
    }


def cosine_similarity_matrix(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Pairwise cosine similarity between embedding matrices (N, D) and (M, D)."""
    a_n = a / a.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    b_n = b / b.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return a_n @ b_n.T


def entropy_bits(probs: torch.Tensor, eps: float = 1e-12) -> float:
    """Shannon entropy in bits for a discrete probability distribution."""
    p = probs.float().clamp(min=eps)
    p = p / p.sum()
    return float(-(p * torch.log2(p)).sum().item())
