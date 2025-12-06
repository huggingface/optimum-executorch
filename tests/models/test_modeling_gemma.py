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

import gc
import logging
import os
import subprocess
import tempfile
import unittest

import pytest
import torchao
from packaging.version import parse
from transformers import AutoTokenizer
from transformers.testing_utils import slow

from executorch.extension.pybindings.portable_lib import ExecuTorchModule
from optimum.executorch import ExecuTorchModelForCausalLM


is_ci = os.environ.get("GITHUB_ACTIONS") == "true"


@pytest.mark.skipif(
    parse(torchao.__version__) < parse("0.11.0.dev0"),
    reason="Only available on torchao >= 0.11.0.dev0",
)
class ExecuTorchModelIntegrationTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @slow
    @pytest.mark.run_slow
    def test_gemma_export_to_executorch(self):
        model_id = "weqweasdas/RM-Gemma-2B"
        task = "text-generation"
        recipe = "xnnpack"
        with tempfile.TemporaryDirectory() as tempdir:
            out_dir = f"{tempdir}/executorch"
            subprocess.run(
                f"optimum-cli export executorch \
                    --model {model_id} \
                    --task {task} \
                    --recipe {recipe} \
                    --output_dir {tempdir}/executorch \
                    --use_custom_sdpa \
                    --qlinear 8da4w \
                    --qembedding 8w",
                shell=True,
                check=True,
            )
            pte_full_path = f"{out_dir}/model.pte"
            self.assertTrue(os.path.exists(pte_full_path))

            # Explicitly delete the PTE file to free up disk space
            if os.path.exists(pte_full_path):
                os.remove(pte_full_path)
            gc.collect()

    @slow
    @pytest.mark.run_slow
    def test_gemma_text_generation_with_custom_sdpa_8da4w_8we(self):
        # TODO: Switch to use google/gemma-2b once https://github.com/huggingface/optimum/issues/2127 is fixed
        # model_id = "google/gemma-2b"
        model_id = "weqweasdas/RM-Gemma-2B"
        # ExecuTorch model + custom sdpa + 8da4w linear quantization + int8 embedding quantization
        kwargs = {"qlinear": "8da4w", "qembedding": "8w"}
        model = ExecuTorchModelForCausalLM.from_pretrained(
            model_id,
            recipe="xnnpack",
            attn_implementation="custom_sdpa",
            **kwargs,
        )
        self.assertIsInstance(model, ExecuTorchModelForCausalLM)
        self.assertIsInstance(model.model, ExecuTorchModule)

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        generated_text = model.text_generation(
            tokenizer=tokenizer,
            prompt="Hello I am doing",
            max_seq_len=21,
        )
        logging.info(f"\nGenerated text:\n\t{generated_text}")

    @slow
    @pytest.mark.run_slow
    @pytest.mark.portable
    @pytest.mark.skipif(is_ci, reason="Too big for CI runners")
    def test_gemma_text_generation_portable(self):
        # TODO: Switch to use google/gemma-2b once https://github.com/huggingface/optimum/issues/2127 is fixed
        # model_id = "google/gemma-2b"
        model_id = "weqweasdas/RM-Gemma-2B"
        model = ExecuTorchModelForCausalLM.from_pretrained(model_id, recipe="portable")
        self.assertIsInstance(model, ExecuTorchModelForCausalLM)
        self.assertIsInstance(model.model, ExecuTorchModule)

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        generated_text = model.text_generation(
            tokenizer=tokenizer,
            prompt="Hello I am doing",
            max_seq_len=21,
        )
        logging.info(f"\nGenerated text:\n\t{generated_text}")
