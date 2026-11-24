"""Comprehensive automated test suite for PyAI-Foundry."""

from __future__ import annotations

import unittest
import torch
from torch.utils.data import TensorDataset

from pyai_foundry import (
    CharTokenizer,
    DenseRetriever,
    MinMaxScaler,
    StandardScaler,
    TrainConfig,
    binary_roc_auc,
    bleu_score,
    bootstrap_confidence_interval,
    build_loss,
    build_model,
    chunk_text,
    classification_report,
    correlation_matrix,
    count_parameters,
    detect_outliers_iqr,
    dice_coefficient,
    entropy_bits,
    inject_lora,
    list_losses,
    list_models,
    mrr_at_k,
    ndcg_at_k,
    perplexity_from_loss,
    psnr,
    regression_report,
    rouge_l,
    run_kfold_cv,
    ssim,
    token_accuracy,
    top_k_top_p_filter,
    train_val_test_split,
)


class TestPyAIFoundry(unittest.TestCase):
    def test_ds_utilities(self) -> None:
        x = torch.randn(40, 5) * 3.0 + 10.0
        xs = StandardScaler().fit_transform(x)
        self.assertLess(abs(xs.mean().item()), 1e-4)
        xm = MinMaxScaler().fit_transform(x)
        self.assertGreaterEqual(xm.min().item(), -1e-5)
        self.assertLessEqual(xm.max().item(), 1.0 + 1e-5)
        tr, va, te = train_val_test_split(40)
        self.assertEqual(len(tr) + len(va) + len(te), 40)
        self.assertEqual(tuple(correlation_matrix(x).shape), (5, 5))
        self.assertEqual(tuple(detect_outliers_iqr(x).shape), (40, 5))
        ci = bootstrap_confidence_interval(x[:, 0], n_bootstrap=100)
        self.assertLessEqual(ci["ci_low"], ci["ci_high"])
        self.assertGreater(entropy_bits(torch.tensor([0.5, 0.5])), 0.99)

    def test_measures_and_losses(self) -> None:
        self.assertGreaterEqual(len(list_losses()), 7)
        preds = torch.tensor([0, 1, 1, 0, 2, 2])
        targets = torch.tensor([0, 1, 0, 0, 2, 2])
        rep = classification_report(preds, targets, num_classes=3)
        self.assertGreater(rep["accuracy"], 0.8)
        self.assertEqual(binary_roc_auc(torch.tensor([0.9, 0.8, 0.2, 0.1]), torch.tensor([1, 1, 0, 0])), 1.0)
        reg = regression_report(torch.tensor([1.0, 2.0, 3.0]), torch.tensor([1.0, 2.0, 3.0]))
        self.assertEqual(reg["r2"], 1.0)
        img = torch.rand(2, 1, 16, 16)
        self.assertGreaterEqual(psnr(img, img), 99.0)
        self.assertGreater(ssim(img, img), 0.99)
        self.assertGreater(dice_coefficient(torch.full((1, 1, 8, 8), 5.0), torch.ones(1, 1, 8, 8)), 0.99)
        self.assertGreater(bleu_score(["deep", "learning", "rocks"], ["deep", "learning", "rocks"]), 0.9)
        self.assertEqual(rouge_l(["a", "b", "c"], ["a", "b", "c"]), 1.0)
        self.assertEqual(mrr_at_k([[0, 1, 0], [1, 0, 0]]), 0.75)
        self.assertEqual(ndcg_at_k([1, 0, 0]), 1.0)
        self.assertAlmostEqual(perplexity_from_loss(0.0), 1.0, places=3)

        # Contrastive InfoNCE & CausalLM losses
        z = torch.randn(4, 16)
        self.assertTrue(torch.isfinite(build_loss("infonce")(z, z)))
        lm_logits = torch.randn(2, 8, 32)
        lm_tgt = torch.randint(0, 32, (2, 8))
        self.assertTrue(torch.isfinite(build_loss("causal_lm")(lm_logits, lm_tgt)))
        self.assertGreaterEqual(token_accuracy(lm_logits, lm_tgt), 0.0)

    def test_architectures_lora_and_llm_utils(self) -> None:
        self.assertGreaterEqual(len(list_models()), 8)
        self.assertEqual(tuple(build_model("tabular_mlp", in_features=8, out_features=3)(torch.randn(4, 8)).shape), (4, 3))
        self.assertEqual(tuple(build_model("resnet_cnn", in_channels=3, num_classes=4, base_channels=8)(torch.randn(2, 3, 32, 32)).shape), (2, 4))
        self.assertEqual(tuple(build_model("unet_2d", in_channels=1, out_channels=1, base_channels=8)(torch.randn(2, 1, 32, 32)).shape), (2, 1, 32, 32))
        self.assertEqual(tuple(build_model("bilstm_attention", input_dim=6, hidden_dim=16, out_features=2)(torch.randn(2, 10, 6)).shape), (2, 2))
        self.assertEqual(tuple(build_model("vit", in_channels=3, image_size=32, patch_size=8, embed_dim=32, depth=2, num_classes=5)(torch.randn(2, 3, 32, 32)).shape), (2, 5))
        self.assertEqual(tuple(build_model("autoencoder", in_features=16, latent_dim=4)(torch.randn(3, 16)).shape), (3, 16))
        za, zb = build_model("dual_encoder", dim_a=12, dim_b=12, proj_dim=8)(torch.randn(3, 12), torch.randn(3, 12))
        self.assertEqual(tuple(za.shape), (3, 8))

        # Modern MiniGPT + LoRA + generation
        gpt = build_model("mini_gpt", vocab_size=64, embed_dim=32, num_layers=2, num_heads=4)
        out = gpt(torch.randint(0, 64, (2, 12)))
        self.assertEqual(tuple(out.shape), (2, 12, 64))
        gen = gpt.generate(torch.randint(0, 64, (1, 4)), max_new_tokens=6)
        self.assertEqual(tuple(gen.shape), (1, 10))

        total_before = count_parameters(gpt, trainable_only=True)
        n_lora = inject_lora(gpt, target_substrings=("q_proj", "v_proj"), r=4)
        self.assertGreater(n_lora, 0)
        self.assertLess(count_parameters(gpt, trainable_only=True), total_before)

        # Tokenizer, chunker, dense retriever, top-p filter
        tok = CharTokenizer("hello world")
        self.assertEqual(tok.decode(tok.encode("hello")), "hello")
        self.assertGreaterEqual(len(chunk_text("one two three four five six", chunk_size=3, overlap=1)), 2)
        ret = DenseRetriever(["doc1", "doc2"], torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
        self.assertEqual(ret.search(torch.tensor([0.9, 0.1]), top_k=1)[0][0], "doc1")
        self.assertEqual(top_k_top_p_filter(torch.randn(10), top_k=3, top_p=0.9).shape, (10,))

    def test_kfold_cv_pipeline(self) -> None:
        x = torch.randn(36, 8)
        y = (x[:, 0] + x[:, 1] > 0).long()
        ds = TensorDataset(x, y)
        res = run_kfold_cv(
            ds,
            model_fn=lambda: build_model("tabular_mlp", in_features=8, out_features=2, hidden_dim=32, depth=2),
            criterion_fn=lambda: build_loss("focal"),
            k=3,
            config=TrainConfig(epochs=3, batch_size=12, use_ema=True),
        )
        self.assertEqual(res["k"], 3)
        self.assertEqual(len(res["folds"]), 3)


if __name__ == "__main__":
    unittest.main()
