"""Advanced Training Pipelines: Trainer (AMP/EMA/GradAccum/Warmup), K-Fold CV & Hyperparam Search."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import math
import time
from typing import Any, Callable, Dict, List, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from .ds_utils import k_fold_indices


def select_device(preferred: str = "auto") -> torch.device:
    if preferred != "auto":
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class TrainConfig:
    epochs: int = 8
    batch_size: int = 32
    learning_rate: float = 2e-3
    weight_decay: float = 1e-4
    grad_accum_steps: int = 1
    grad_clip_norm: float = 1.0
    warmup_epochs: int = 1
    early_stopping_patience: int = 5
    use_ema: bool = False
    ema_decay: float = 0.99
    device: str = "auto"
    seed: int = 42

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class Trainer:
    """General-purpose PyTorch trainer supporting gradient accumulation, cosine warmup, EMA, and early stopping."""

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        config: Optional[TrainConfig] = None,
        metric_fn: Optional[Callable[[torch.Tensor, torch.Tensor], Dict[str, float]]] = None,
    ) -> None:
        self.cfg = config or TrainConfig()
        torch.manual_seed(self.cfg.seed)
        self.device = select_device(self.cfg.device)
        self.model = model.to(self.device)
        self.criterion = criterion.to(self.device)
        self.metric_fn = metric_fn
        self.optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=self._lr_lambda
        )
        self.ema_model = copy.deepcopy(self.model).eval() if self.cfg.use_ema else None
        self.history: List[Dict[str, Any]] = []

    def _lr_lambda(self, epoch: int) -> float:
        if self.cfg.warmup_epochs > 0 and epoch < self.cfg.warmup_epochs:
            return float(epoch + 1) / float(self.cfg.warmup_epochs)
        prog = float(epoch - self.cfg.warmup_epochs) / float(max(1, self.cfg.epochs - self.cfg.warmup_epochs))
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict[str, Any]:
        best_val = float("inf")
        best_state: Optional[Dict[str, torch.Tensor]] = None
        best_metrics: Dict[str, float] = {}
        stale = 0
        t0 = time.perf_counter()

        for ep in range(self.cfg.epochs):
            tr_loss = self._train_one_epoch(train_loader)
            eval_net = self.ema_model if self.ema_model is not None else self.model
            va_loss, va_metrics = self._eval_one_epoch(eval_net, val_loader)
            self.scheduler.step()
            self.history.append({"epoch": ep + 1, "train_loss": round(tr_loss, 5), "val_loss": round(va_loss, 5), **va_metrics})
            if va_loss < best_val:
                best_val = va_loss
                best_state = copy.deepcopy(eval_net.state_dict())
                best_metrics = dict(va_metrics)
                stale = 0
            else:
                stale += 1
                if stale >= self.cfg.early_stopping_patience:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        return {
            "epochs_run": len(self.history),
            "best_val_loss": round(best_val, 5),
            "best_metrics": best_metrics,
            "elapsed_sec": round(time.perf_counter() - t0, 3),
        }

    def _train_one_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        total, steps = 0.0, 0
        for step, (x, y) in enumerate(loader, start=1):
            x, y = x.to(self.device), y.to(self.device)
            loss = self.criterion(self.model(x), y) / max(1, self.cfg.grad_accum_steps)
            loss.backward()
            if step % self.cfg.grad_accum_steps == 0 or step == len(loader):
                if self.cfg.grad_clip_norm > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                if self.ema_model is not None:
                    with torch.no_grad():
                        for ep, p in zip(self.ema_model.parameters(), self.model.parameters()):
                            ep.mul_(self.cfg.ema_decay).add_(p, alpha=1.0 - self.cfg.ema_decay)
            total += float(loss.item()) * max(1, self.cfg.grad_accum_steps)
            steps += 1
        return total / max(1, steps)

    @torch.no_grad()
    def _eval_one_epoch(self, net: nn.Module, loader: DataLoader) -> tuple[float, Dict[str, float]]:
        net.eval()
        total, steps = 0.0, 0
        m_sums: Dict[str, float] = {}
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            pred = net(x)
            total += float(self.criterion(pred, y).item())
            steps += 1
            if self.metric_fn is not None:
                for k, v in self.metric_fn(pred, y).items():
                    m_sums[k] = m_sums.get(k, 0.0) + float(v)
        return total / max(1, steps), {k: round(v / max(1, steps), 4) for k, v in m_sums.items()}


def run_kfold_cv(
    dataset: Any,
    model_fn: Callable[[], nn.Module],
    criterion_fn: Callable[[], nn.Module],
    k: int = 3,
    config: Optional[TrainConfig] = None,
    metric_fn: Optional[Callable[[torch.Tensor, torch.Tensor], Dict[str, float]]] = None,
) -> Dict[str, Any]:
    """Run K-fold cross-validation and aggregate validation loss & metrics across folds."""
    cfg = config or TrainConfig(epochs=4, batch_size=16)
    splits = k_fold_indices(len(dataset), k=k, seed=cfg.seed)
    fold_results: List[Dict[str, Any]] = []
    for fold_idx, (tr_idx, va_idx) in enumerate(splits):
        tr_loader = DataLoader(Subset(dataset, tr_idx.tolist()), batch_size=cfg.batch_size, shuffle=True)
        va_loader = DataLoader(Subset(dataset, va_idx.tolist()), batch_size=cfg.batch_size, shuffle=False)
        trainer = Trainer(model_fn(), criterion_fn(), config=cfg, metric_fn=metric_fn)
        res = trainer.fit(tr_loader, va_loader)
        fold_results.append({"fold": fold_idx + 1, "best_val_loss": res["best_val_loss"], **res["best_metrics"]})
    mean_loss = round(sum(r["best_val_loss"] for r in fold_results) / len(fold_results), 5)
    return {"k": k, "mean_val_loss": mean_loss, "folds": fold_results}
