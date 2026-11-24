# PyAI-Foundry

An **exclusively Python** all-in-one toolkit spanning **Data Science, Machine Learning, Deep Learning, Computer Vision, and Large Language Models (LLMs)**—designed so every measure, loss, preprocessing utility, neural architecture, and training pipeline can be imported and called cleanly in one line.

## Module Map

| Module | Highlights |
| :--- | :--- |
| `pyai_foundry.ds_utils` | `StandardScaler`, `MinMaxScaler`, `train_val_test_split`, `k_fold_indices`, `detect_outliers_iqr`, `correlation_matrix`, `bootstrap_confidence_interval`, `cosine_similarity_matrix`, `entropy_bits` |
| `pyai_foundry.metrics` | Classification (`confusion_matrix`, `classification_report`, `binary_roc_auc`), Regression (`regression_report`: MSE/RMSE/MAE/R2/MAPE), Vision (`psnr`, `ssim`, `dice_coefficient`), LLM & RAG (`perplexity_from_loss`, `token_accuracy`, `bleu_score`, `rouge_l`, `mrr_at_k`, `ndcg_at_k`) |
| `pyai_foundry.losses` | Callable registry (`build_loss`): `focal`, `label_smoothing_ce`, `charbonnier`, `dice`, `infonce`, `causal_lm`, `huber` |
| `pyai_foundry.architectures` | Callable zoo (`build_model`): `tabular_mlp`, `resnet_cnn`, `unet_2d`, `bilstm_attention`, `vit`, `mini_gpt` (RMSNorm + RoPE + SwiGLU + top-k generation), `autoencoder`, `dual_encoder`, plus `LoRALinear` & `inject_lora()` PEFT helper |
| `pyai_foundry.llm_utils` | `CharTokenizer`, `chunk_text`, `DenseRetriever` (cosine top-k vector index), `top_k_top_p_filter` (nucleus sampling) |
| `pyai_foundry.trainer` | `Trainer` (auto CUDA/MPS/CPU, cosine warmup LR, gradient accumulation, gradient clipping, EMA, early stopping) and `run_kfold_cv` |

## Quickstart

```python
import torch
from pyai_foundry import (
    StandardScaler,
    TrainConfig,
    build_loss,
    build_model,
    inject_lora,
    list_models,
    run_kfold_cv,
)

# 1. Inspect all registered DL & LLM architectures
print(list_models())

# 2. Build a modern decoder-only LLM (RMSNorm + RoPE + SwiGLU) and apply LoRA
gpt = build_model("mini_gpt", vocab_size=256, embed_dim=128, num_layers=3, num_heads=4)
inject_lora(gpt, target_substrings=("q_proj", "v_proj"), r=4)

# 3. Generate tokens
prompt = torch.randint(0, 256, (1, 8))
generated = gpt.generate(prompt, max_new_tokens=16, temperature=0.8, top_k=16)
```

## Run Tests & CLI

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
PYTHONPATH=src python3 -m pyai_foundry catalog
```
