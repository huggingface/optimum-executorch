# coding=utf-8
# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

import pytest
from executorch.extension.pybindings.portable_lib import ExecuTorchModule
from transformers import AutoTokenizer
from transformers.testing_utils import slow

from optimum.executorch import ExecuTorchModelForMaskedLM


class ExecuTorchModelIntegrationTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @slow
    @pytest.mark.run_slow
    def test_eurobert_fill_mask_to_executorch(self):
        model_id = "EuroBERT/EuroBERT-210m"
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        # Test fetching and lowering the model to ExecuTorch
        model = ExecuTorchModelForMaskedLM.from_pretrained(
            model_id=model_id,
            recipe="xnnpack",
            trust_remote_code=True,  # This model is on Hub only
        )
        self.assertIsInstance(model, ExecuTorchModelForMaskedLM)
        self.assertIsInstance(model.model, ExecuTorchModule)

        test_text = f"Paris is the {tokenizer.mask_token} of France."
        inputs = tokenizer(
            test_text,
            return_tensors="pt",
            padding="max_length",
            max_length=10,
        )

        # Test inference using ExecuTorch model
        exported_outputs = model.forward(inputs["input_ids"], inputs["attention_mask"])
        predicted_masks = tokenizer.decode(exported_outputs[0, 4].topk(5).indices)
        self.assertTrue(
            any(word in predicted_masks for word in ["capital", "center", "heart", "birthplace"]),
            f"Exported model predictions {predicted_masks} don't contain any of the most common expected words",
        )
