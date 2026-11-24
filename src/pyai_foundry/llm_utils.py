"""LLM & RAG Supporting Utilities: Tokenization, Chunking, Dense Retrieval, & Sampling."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple
import torch


class CharTokenizer:
    """Deterministic character-level tokenizer for lightweight LLM experiments."""

    def __init__(self, text: str) -> None:
        chars = sorted(set(text))
        self.stoi: Dict[str, int] = {ch: i for i, ch in enumerate(chars)}
        self.itos: Dict[int, str] = {i: ch for i, ch in enumerate(chars)}

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def encode(self, text: str) -> List[int]:
        return [self.stoi[c] for c in text if c in self.stoi]

    def decode(self, ids: Sequence[int]) -> str:
        return "".join(self.itos[i] for i in ids if i in self.itos)


def chunk_text(text: str, chunk_size: int = 200, overlap: int = 40) -> List[str]:
    """Split text into overlapping word-boundary chunks for RAG indexing."""
    words = text.split()
    if not words:
        return []
    step = max(1, chunk_size - overlap)
    return [" ".join(words[i : i + chunk_size]) for i in range(0, len(words), step)]


class DenseRetriever:
    """In-memory cosine-similarity vector index for RAG top-k retrieval."""

    def __init__(self, documents: Sequence[str], embeddings: torch.Tensor) -> None:
        self.documents = list(documents)
        self.embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    @torch.no_grad()
    def search(self, query_emb: torch.Tensor, top_k: int = 3) -> List[Tuple[str, float]]:
        q = query_emb.float().flatten()
        q = q / q.norm().clamp(min=1e-8)
        scores = self.embeddings @ q
        vals, idxs = torch.topk(scores, k=min(top_k, len(self.documents)))
        return [(self.documents[i], round(float(v), 4)) for v, i in zip(vals.tolist(), idxs.tolist())]


def top_k_top_p_filter(logits: torch.Tensor, top_k: int = 0, top_p: float = 1.0) -> torch.Tensor:
    """Filter logits using top-k and/or nucleus (top-p) sampling thresholds."""
    filtered = logits.clone()
    if top_k > 0:
        v, _ = torch.topk(filtered, min(top_k, filtered.size(-1)))
        filtered[filtered < v[..., [-1]]] = float("-inf")
    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(filtered, descending=True, dim=-1)
        cum_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        remove = cum_probs > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        scatter_mask = torch.zeros_like(remove).scatter(-1, sorted_indices, remove)
        filtered[scatter_mask] = float("-inf")
    return filtered
