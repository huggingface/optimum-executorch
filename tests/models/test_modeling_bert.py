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

import logging
import os
import subprocess
import tempfile
import unittest

import pytest
import torchao
from executorch.extension.pybindings.portable_lib import ExecuTorchModule

from optimum.executorch import ExecuTorchModelForMaskedLM
from packaging.version import parse
from transformers import AutoTokenizer
from transformers.testing_utils import slow


@pytest.mark.skipif(
    parse(torchao.__version__) < parse("0.11.0.dev0"),
    reason="Only available on torchao >= 0.11.0.dev0",
)
class ExecuTorchModelIntegrationTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @slow
    @pytest.mark.run_slow
    def test_bert_export_to_executorch(self):
        model_id = "google-bert/bert-base-uncased"
        task = "fill-mask"
        recipe = "xnnpack"
        with tempfile.TemporaryDirectory() as tempdir:
            subprocess.run(
                f"optimum-cli export executorch --model {model_id} --task {task} --recipe {recipe} --output_dir {tempdir}/executorch",
                shell=True,
                check=True,
            )
            self.assertTrue(os.path.exists(f"{tempdir}/executorch/model.pte"))

    @slow
    @pytest.mark.run_slow
    def test_bert_export_to_executorch_quantized(self):
        model_id = "google-bert/bert-base-uncased"
        task = "fill-mask"
        recipe = "xnnpack"
        with tempfile.TemporaryDirectory() as tempdir:
            subprocess.run(
                f"optimum-cli export executorch --model {model_id} --task {task} --recipe {recipe} --qlinear 8da4w --output_dir {tempdir}/executorch",
                shell=True,
                check=True,
            )
            self.assertTrue(os.path.exists(f"{tempdir}/executorch/model.pte"))

    def _helper_bert_fill_mask(self, recipe: str):
        model_id = "google-bert/bert-base-uncased"
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        # Test fetching and lowering the model to ExecuTorch
        model = ExecuTorchModelForMaskedLM.from_pretrained(
            model_id=model_id, recipe=recipe
        )
        self.assertIsInstance(model, ExecuTorchModelForMaskedLM)
        self.assertIsInstance(model.model, ExecuTorchModule)

        input_text = f"Paris is the {tokenizer.mask_token} of France."
        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            padding="max_length",
            max_length=10,
        )

        # Test inference using ExecuTorch model
        exported_outputs = model.forward(inputs["input_ids"], inputs["attention_mask"])
        predicted_masks = tokenizer.decode(exported_outputs[0, 4].topk(5).indices)
        logging.info(
            f"\nInput text:\n\t{input_text}\nPredicted masks:\n\t{predicted_masks}"
        )
        self.assertTrue(
            any(
                word in predicted_masks
                for word in ["capital", "center", "heart", "birthplace"]
            ),
            f"Exported model predictions {predicted_masks} don't contain any of the most common expected words",
        )

    @slow
    @pytest.mark.run_slow
    def test_bert_fill_mask(self):
        self._helper_bert_fill_mask("xnnpack")

    @slow
    @pytest.mark.run_slow
    @pytest.mark.portable
    def test_bert_fill_mask_portable(self):
        self._helper_bert_fill_mask("portable")

    @pytest.mark.run_slow
    def test_bert_fill_mask_qnn(self):
        self._helper_bert_fill_mask(recipe="qnn_fp16_SM8650")
