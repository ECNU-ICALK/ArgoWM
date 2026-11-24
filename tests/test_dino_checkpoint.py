from __future__ import annotations

from pathlib import Path
import sys
import unittest

import torch
from timm.models.vision_transformer import VisionTransformer
from transformers import Dinov2Config, Dinov2Model

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wgsr_v9.dino_checkpoint import hf_dinov2_to_timm


class DinoCheckpointTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(41)
        self.hf = Dinov2Model(Dinov2Config(
            image_size=28, patch_size=14, hidden_size=32,
            num_hidden_layers=2, num_attention_heads=4, mlp_ratio=2,
        )).eval()

    def test_converted_encoder_matches_huggingface_cls_on_identical_input(self) -> None:
        model = VisionTransformer(
            img_size=28, patch_size=14, embed_dim=32, depth=2,
            num_heads=4, mlp_ratio=2, qkv_bias=True, init_values=1.0,
            num_classes=0,
        ).eval()
        inputs = torch.randn(3, 3, 28, 28)
        with torch.inference_mode():
            expected = self.hf(inputs).last_hidden_state[:, 0]
        legacy = {}
        for key, value in self.hf.state_dict().items():
            for current, old in (("q_proj", "attention.query"), ("k_proj", "attention.key"),
                                 ("v_proj", "attention.value"), ("o_proj", "output.dense")):
                key = key.replace(f"attention.{current}.", f"attention.{old}.")
            legacy[key] = value
        for state in (self.hf.state_dict(), legacy):
            model.load_state_dict(hf_dinov2_to_timm(state), strict=True)
            with torch.inference_mode():
                actual = model.forward_head(model.forward_features(inputs), pre_logits=True)
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=2e-6)

    def test_rejects_unsupported_or_missing_source_weights(self) -> None:
        state = dict(self.hf.state_dict())
        state["unknown_tensor"] = torch.zeros(1)
        with self.assertRaisesRegex(ValueError, "unsupported"):
            hf_dinov2_to_timm(state)
        del state["unknown_tensor"]
        key = next(key for key in state if key.endswith(("attention.key.weight", "attention.k_proj.weight")))
        del state[key]
        with self.assertRaises(KeyError):
            hf_dinov2_to_timm(state)


if __name__ == "__main__":
    unittest.main()
