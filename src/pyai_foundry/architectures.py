"""Unified Deep Learning & LLM Architecture Zoo (One-Line Callable Registry).

Includes:
- tabular_mlp: Residual Tabular MLP with LayerNorm + GELU
- resnet_cnn: Multi-stage Residual 2D CNN
- unet_2d: Skip-connected 2D U-Net
- bilstm_attention: Bidirectional LSTM + Additive Self-Attention
- vit: Vision Transformer (Patch Embedding + Pre-Norm MHSA)
- mini_gpt: Modern Decoder-Only LLM (RMSNorm + RoPE + SwiGLU + Causal Attention + Top-K Generation)
- autoencoder: Bottleneck Representation / Anomaly Detection Autoencoder
- dual_encoder: Contrastive Dual-Branch Encoder (InfoNCE-ready)
- LoRALinear + inject_lora(): Parameter-Efficient Fine-Tuning (PEFT) helper
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

_MODEL_REGISTRY: Dict[str, Callable[..., nn.Module]] = {}


def register_model(name: str) -> Callable[[Callable[..., nn.Module]], Callable[..., nn.Module]]:
    key = name.strip().lower()

    def decorator(fn: Callable[..., nn.Module]) -> Callable[..., nn.Module]:
        _MODEL_REGISTRY[key] = fn
        return fn

    return decorator


def list_models() -> List[str]:
    return sorted(_MODEL_REGISTRY.keys())


def build_model(name: str, **kwargs: Any) -> nn.Module:
    key = name.strip().lower()
    if key not in _MODEL_REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Available: {', '.join(list_models())}")
    return _MODEL_REGISTRY[key](**kwargs)


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


# ==================== PEFT: LoRA Adapter ====================

class LoRALinear(nn.Module):
    """Drop-in Low-Rank Adaptation (LoRA) wrapper around any frozen nn.Linear layer."""

    def __init__(self, base_layer: nn.Linear, r: int = 4, alpha: float = 8.0) -> None:
        super().__init__()
        self.base = base_layer
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.r = r
        self.scaling = alpha / max(1, r)
        self.lora_a = nn.Parameter(torch.randn(r, base_layer.in_features) * 0.02)
        self.lora_b = nn.Parameter(torch.zeros(base_layer.out_features, r))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + (x @ self.lora_a.T @ self.lora_b.T) * self.scaling


def inject_lora(
    model: nn.Module,
    target_substrings: Tuple[str, ...] = ("q_proj", "v_proj", "fc1", "fc2"),
    r: int = 4,
    alpha: float = 8.0,
) -> int:
    """Freeze base weights and replace matching Linear layers with LoRALinear."""
    for p in model.parameters():
        p.requires_grad_(False)
    replaced = 0
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            full_name = f"{name}.{child_name}" if name else child_name
            if isinstance(child, nn.Linear) and any(t in full_name for t in target_substrings):
                setattr(module, child_name, LoRALinear(child, r=r, alpha=alpha))
                replaced += 1
    return replaced


# ==================== 1. Tabular Residual MLP ====================

@register_model("tabular_mlp")
class ResidualTabularMLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_dim: int = 128,
        depth: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.in_proj = nn.Sequential(nn.Linear(in_features, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                for _ in range(depth)
            ]
        )
        self.head = nn.Linear(hidden_dim, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x)
        for blk in self.blocks:
            h = h + blk(h)
        return self.head(h)


# ==================== 2. Residual 2D CNN ====================

class _ResBlock2d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.skip = nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.conv(x) + self.skip(x))


@register_model("resnet_cnn")
class ResNetCNN(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 10, base_channels: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            _ResBlock2d(in_channels, base_channels),
            nn.Conv2d(base_channels, base_channels * 2, 3, stride=2, padding=1),
            _ResBlock2d(base_channels * 2, base_channels * 2),
            nn.Conv2d(base_channels * 2, base_channels * 4, 3, stride=2, padding=1),
            _ResBlock2d(base_channels * 4, base_channels * 4),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(base_channels * 4, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ==================== 3. 2D U-Net ====================

@register_model("unet_2d")
class UNet2D(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 1, base_channels: int = 24) -> None:
        super().__init__()
        self.enc1 = _ResBlock2d(in_channels, base_channels)
        self.enc2 = _ResBlock2d(base_channels, base_channels * 2)
        self.bot = _ResBlock2d(base_channels * 2, base_channels * 4)
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.dec2 = _ResBlock2d(base_channels * 4, base_channels * 2)
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
        self.dec1 = _ResBlock2d(base_channels * 2, base_channels)
        self.head = nn.Conv2d(base_channels, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        b = self.bot(F.max_pool2d(e2, 2))
        d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)


# ==================== 4. BiLSTM + Self-Attention ====================

@register_model("bilstm_attention")
class BiLSTMAttention(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, out_features: int = 2
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True
        )
        self.attn_score = nn.Linear(hidden_dim * 2, 1)
        self.head = nn.Linear(hidden_dim * 2, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_out, _ = self.lstm(x)
        weights = torch.softmax(self.attn_score(seq_out), dim=1)
        context = (seq_out * weights).sum(dim=1)
        return self.head(context)


# ==================== 5. Vision Transformer (ViT) ====================

@register_model("vit")
class VisionTransformer(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        image_size: int = 32,
        patch_size: int = 8,
        embed_dim: int = 96,
        depth: int = 3,
        num_heads: int = 4,
        num_classes: int = 10,
    ) -> None:
        super().__init__()
        n_patches = (image_size // patch_size) ** 2
        self.patch = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos = nn.Parameter(torch.randn(1, n_patches + 1, embed_dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 2, activation="gelu", batch_first=True
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=depth)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = self.patch(x).flatten(2).transpose(1, 2)
        t = torch.cat([self.cls.expand(x.size(0), -1, -1), t], dim=1) + self.pos[:, : t.size(1) + 1]
        return self.head(self.enc(t)[:, 0])


# ==================== 6. Modern Decoder-Only LLM (RMSNorm + RoPE + SwiGLU) ====================

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * rms * self.weight


def _apply_rope(x: torch.Tensor) -> torch.Tensor:
    """Apply Rotary Position Embeddings to (B, H, T, D)."""
    _, _, t, d = x.shape
    half = d // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(0, half, device=x.device, dtype=x.dtype) / max(1, half)
    )
    pos = torch.arange(t, device=x.device, dtype=x.dtype)
    angles = torch.outer(pos, freqs).unsqueeze(0).unsqueeze(0)
    cos, sin = torch.cos(angles), torch.sin(angles)
    x1, x2 = x[..., :half], x[..., half : half * 2]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class _ModernLLMBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.norm1 = RMSNorm(dim)
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        self.norm2 = RMSNorm(dim)
        hidden = int(dim * 8 / 3)
        self.gate_proj = nn.Linear(dim, hidden, bias=False)
        self.up_proj = nn.Linear(dim, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        h = self.norm1(x)
        q = _apply_rope(self.q_proj(h).view(b, t, self.num_heads, self.head_dim).transpose(1, 2))
        k = _apply_rope(self.k_proj(h).view(b, t, self.num_heads, self.head_dim).transpose(1, 2))
        v = self.v_proj(h).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn_out = attn_out.transpose(1, 2).contiguous().view(b, t, c)
        x = x + self.o_proj(attn_out)
        h2 = self.norm2(x)
        x = x + self.down_proj(F.silu(self.gate_proj(h2)) * self.up_proj(h2))
        return x


@register_model("mini_gpt")
class MiniGPT(nn.Module):
    """Decoder-only Transformer LLM with RMSNorm, RoPE, SwiGLU, and autoregressive generation."""

    def __init__(
        self,
        vocab_size: int = 256,
        embed_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
        max_seq_len: int = 128,
    ) -> None:
        super().__init__()
        self.max_seq_len = max_seq_len
        self.tok_emb = nn.Embedding(vocab_size, embed_dim)
        self.blocks = nn.ModuleList([_ModernLLMBlock(embed_dim, num_heads) for _ in range(num_layers)])
        self.norm = RMSNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        h = self.tok_emb(idx)
        for blk in self.blocks:
            h = blk(h)
        return self.lm_head(self.norm(h))

    @torch.no_grad()
    def generate(
        self, idx: torch.Tensor, max_new_tokens: int = 16, temperature: float = 1.0, top_k: int = 16
    ) -> torch.Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            ctx = idx[:, -self.max_seq_len :]
            logits = self(ctx)[:, -1, :] / max(1e-5, temperature)
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = torch.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_tok], dim=1)
        return idx


# ==================== 7. Autoencoder & 8. Contrastive Dual Encoder ====================

@register_model("autoencoder")
class DenseAutoencoder(nn.Module):
    def __init__(self, in_features: int = 64, latent_dim: int = 16) -> None:
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(in_features, latent_dim * 2), nn.GELU(), nn.Linear(latent_dim * 2, latent_dim))
        self.decoder = nn.Sequential(nn.Linear(latent_dim, latent_dim * 2), nn.GELU(), nn.Linear(latent_dim * 2, in_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


@register_model("dual_encoder")
class ContrastiveDualEncoder(nn.Module):
    def __init__(self, dim_a: int = 64, dim_b: int = 64, proj_dim: int = 32) -> None:
        super().__init__()
        self.enc_a = nn.Sequential(nn.Linear(dim_a, proj_dim * 2), nn.GELU(), nn.Linear(proj_dim * 2, proj_dim))
        self.enc_b = nn.Sequential(nn.Linear(dim_b, proj_dim * 2), nn.GELU(), nn.Linear(proj_dim * 2, proj_dim))

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return F.normalize(self.enc_a(a), dim=-1), F.normalize(self.enc_b(b), dim=-1)
