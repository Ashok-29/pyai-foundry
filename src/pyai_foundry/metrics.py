"""Unified Evaluation Measures across DS, Classical ML, Deep Learning, Vision, Retrieval & LLMs."""

from __future__ import annotations

from collections import Counter
import math
from typing import Dict, List, Sequence
import torch
import torch.nn.functional as F


# ==================== Classification Measures ====================

@torch.no_grad()
def confusion_matrix(preds: torch.Tensor, targets: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Compute (num_classes, num_classes) confusion matrix [true, pred]."""
    cm = torch.zeros((num_classes, num_classes), dtype=torch.long)
    for t, p in zip(targets.view(-1).tolist(), preds.view(-1).tolist()):
        cm[int(t), int(p)] += 1
    return cm


@torch.no_grad()
def classification_report(
    preds: torch.Tensor, targets: torch.Tensor, num_classes: int
) -> Dict[str, float]:
    """Compute accuracy, macro precision, macro recall, and macro F1."""
    cm = confusion_matrix(preds, targets, num_classes).float()
    tp = cm.diag()
    fp = cm.sum(dim=0) - tp
    fn = cm.sum(dim=1) - tp
    precision = (tp / (tp + fp).clamp(min=1e-8)).mean().item()
    recall = (tp / (tp + fn).clamp(min=1e-8)).mean().item()
    f1 = (2 * tp / (2 * tp + fp + fn).clamp(min=1e-8)).mean().item()
    acc = (tp.sum() / cm.sum().clamp(min=1e-8)).item()
    return {
        "accuracy": round(acc, 4),
        "macro_precision": round(precision, 4),
        "macro_recall": round(recall, 4),
        "macro_f1": round(f1, 4),
    }


@torch.no_grad()
def binary_roc_auc(scores: torch.Tensor, targets: torch.Tensor) -> float:
    """Exact trapezoidal ROC-AUC for binary classification scores."""
    s = scores.float().flatten()
    y = targets.float().flatten()
    order = torch.argsort(s, descending=True)
    y_sorted = y[order]
    pos = y_sorted.sum().item()
    neg = (1.0 - y_sorted).sum().item()
    if pos == 0 or neg == 0:
        return 0.5
    tps = torch.cumsum(y_sorted, dim=0)
    fps = torch.cumsum(1.0 - y_sorted, dim=0)
    tpr = torch.cat([torch.tensor([0.0]), tps / pos])
    fpr = torch.cat([torch.tensor([0.0]), fps / neg])
    return round(float(torch.trapezoid(tpr, fpr).item()), 4)


# ==================== Regression Measures ====================

@torch.no_grad()
def regression_report(preds: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    """Compute MSE, RMSE, MAE, R2, and MAPE."""
    p = preds.float().flatten()
    t = targets.float().flatten()
    mse = F.mse_loss(p, t).item()
    mae = F.l1_loss(p, t).item()
    ss_res = ((t - p) ** 2).sum().item()
    ss_tot = ((t - t.mean()) ** 2).sum().item()
    r2 = 1.0 - ss_res / max(1e-8, ss_tot)
    mape = (torch.abs((t - p) / t.abs().clamp(min=1e-6)).mean().item()) * 100.0
    return {
        "mse": round(mse, 5),
        "rmse": round(math.sqrt(mse), 5),
        "mae": round(mae, 5),
        "r2": round(r2, 4),
        "mape_pct": round(mape, 3),
    }


# ==================== Computer Vision Measures ====================

@torch.no_grad()
def psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> float:
    mse = F.mse_loss(pred, target).item()
    return 100.0 if mse <= 1e-12 else round(10.0 * math.log10((data_range**2) / mse), 4)


@torch.no_grad()
def ssim(pred: torch.Tensor, target: torch.Tensor, window_size: int = 7) -> float:
    pad = window_size // 2
    c1, c2 = 0.01**2, 0.03**2
    mu_x = F.avg_pool2d(pred, window_size, stride=1, padding=pad)
    mu_y = F.avg_pool2d(target, window_size, stride=1, padding=pad)
    sx = F.avg_pool2d(pred * pred, window_size, stride=1, padding=pad) - mu_x * mu_x
    sy = F.avg_pool2d(target * target, window_size, stride=1, padding=pad) - mu_y * mu_y
    sxy = F.avg_pool2d(pred * target, window_size, stride=1, padding=pad) - mu_x * mu_y
    val = ((2 * mu_x * mu_y + c1) * (2 * sxy + c2)) / (
        (mu_x * mu_x + mu_y * mu_y + c1) * (sx + sy + c2) + 1e-8
    )
    return round(float(val.mean().item()), 4)


@torch.no_grad()
def dice_coefficient(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    preds = (torch.sigmoid(logits) >= threshold).float()
    inter = (preds * target).sum()
    denom = preds.sum() + target.sum()
    return round(float(((2.0 * inter + 1e-6) / (denom + 1e-6)).item()), 4)


# ==================== LLM, NLP & Retrieval Measures ====================

@torch.no_grad()
def perplexity_from_loss(cross_entropy_loss: float) -> float:
    """Compute language-model perplexity exp(CE loss)."""
    return round(math.exp(min(20.0, cross_entropy_loss)), 4)


@torch.no_grad()
def token_accuracy(logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -100) -> float:
    """Next-token prediction accuracy ignoring padding tokens."""
    preds = logits.argmax(dim=-1)
    mask = targets != ignore_index
    if mask.sum().item() == 0:
        return 0.0
    return round(float((preds[mask] == targets[mask]).float().mean().item()), 4)


def bleu_score(hypothesis: Sequence[str], reference: Sequence[str], max_n: int = 4) -> float:
    """Smoothed sentence BLEU-N score (0..1)."""
    if not hypothesis or not reference:
        return 0.0
    eff_n = min(max_n, len(hypothesis), len(reference))
    precisions: List[float] = []
    for n in range(1, eff_n + 1):
        hyp_ngrams = Counter(tuple(hypothesis[i : i + n]) for i in range(len(hypothesis) - n + 1))
        ref_ngrams = Counter(tuple(reference[i : i + n]) for i in range(len(reference) - n + 1))
        overlap = sum(min(c, ref_ngrams[g]) for g, c in hyp_ngrams.items())
        total = max(1, sum(hyp_ngrams.values()))
        precisions.append((overlap + 1.0) / (total + 1.0))
    bp = 1.0 if len(hypothesis) >= len(reference) else math.exp(1.0 - len(reference) / len(hypothesis))
    score = bp * math.exp(sum(math.log(p) for p in precisions) / eff_n)
    return round(score, 4)


def rouge_l(hypothesis: Sequence[str], reference: Sequence[str]) -> float:
    """ROUGE-L F1 based on Longest Common Subsequence."""
    m, n = len(hypothesis), len(reference)
    if m == 0 or n == 0:
        return 0.0
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if hypothesis[i - 1] == reference[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[m][n]
    p, r = lcs / m, lcs / n
    return round((2 * p * r / (p + r)) if (p + r) > 0 else 0.0, 4)


def mrr_at_k(ranked_relevance: Sequence[Sequence[int]], k: int = 10) -> float:
    """Mean Reciprocal Rank @ K for RAG / retrieval evaluation."""
    rr_sum = 0.0
    for rels in ranked_relevance:
        rr = 0.0
        for rank, rel in enumerate(rels[:k], start=1):
            if rel > 0:
                rr = 1.0 / rank
                break
        rr_sum += rr
    return round(rr_sum / max(1, len(ranked_relevance)), 4)


def ndcg_at_k(ranked_relevance: Sequence[int], k: int = 10) -> float:
    """Normalized Discounted Cumulative Gain @ K."""
    rel = list(ranked_relevance[:k])
    dcg = sum((2**r - 1) / math.log2(idx + 2) for idx, r in enumerate(rel))
    ideal = sorted(rel, reverse=True)
    idcg = sum((2**r - 1) / math.log2(idx + 2) for idx, r in enumerate(ideal))
    return round(dcg / idcg, 4) if idcg > 0 else 0.0
