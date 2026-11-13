import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import save_file

from omni_speech.evaluate.merge_stage1_with_base import interpolate_model, weight_map


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layer = nn.Linear(2, 2, bias=False)
        self.model.projector = nn.Linear(2, 2, bias=False)  # Stage 1 only: left alone


class MergeTest(unittest.TestCase):
    def test_single_file_base_with_qwen_names(self):
        # Qwen3.5 text checkpoints are one file, with weights saved as model.language_model.*.
        model = Tiny()
        torch.nn.init.constant_(model.model.layer.weight, 1.0)
        torch.nn.init.constant_(model.model.projector.weight, 5.0)
        with tempfile.TemporaryDirectory() as base:
            save_file({"model.language_model.layer.weight": torch.full((2, 2), 3.0)},
                      str(Path(base) / "model.safetensors"))
            self.assertEqual(weight_map(base), {"model.language_model.layer.weight": "model.safetensors"})
            interpolate_model(model, base, 0.25)
        self.assertTrue(torch.allclose(model.model.layer.weight, torch.full((2, 2), 0.25 * 1 + 0.75 * 3)))
        self.assertTrue(torch.allclose(model.model.projector.weight, torch.full((2, 2), 5.0)))


if __name__ == "__main__":
    unittest.main()
