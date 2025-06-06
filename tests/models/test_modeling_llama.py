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
from executorch.extension.pybindings.portable_lib import ExecuTorchModule
from packaging.version import parse
from transformers import AutoTokenizer
from transformers.testing_utils import slow

from optimum.executorch import ExecuTorchModelForCausalLM

from ..utils import check_causal_lm_output_quality


@pytest.mark.skipif(
    parse(torchao.__version__) < parse("0.11.0.dev0"),
    reason="Only available on torchao >= 0.11.0.dev0",
)
class ExecuTorchModelIntegrationTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @slow
    @pytest.mark.run_slow
    def test_llama3_2_1b_export_to_executorch(self):
        model_id = "NousResearch/Llama-3.2-1B"
        task = "text-generation"
        recipe = "xnnpack"
        with tempfile.TemporaryDirectory() as tempdir:
            out_dir = f"{tempdir}/executorch"
            subprocess.run(
                f"optimum-cli export executorch \
                    --model {model_id} \
                    --task {task} \
                    --recipe {recipe} \
                    --use_custom_sdpa \
                    --use_custom_kv_cache \
                    --qlinear \
                    --qembedding \
                    --output_dir {tempdir}/executorch",
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
    def test_llama_text_generation_with_custom_sdpa_8da4w_8we(self):
        # ExecuTorch model + custom sdpa + 8da4w linear quantization + int8 embedding quantization
        model_id = "NousResearch/Llama-3.2-1B"
        kwargs = {"qlinear": True, "qembedding": True}
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
            prompt="Simply put, the theory of relativity states that",
            max_seq_len=32,
        )
        logging.info(f"\nGenerated text:\n\t{generated_text}")
        generated_tokens = tokenizer(generated_text, return_tensors="pt").input_ids

        # Free memory before loading eager for quality check
        del model
        del tokenizer
        gc.collect()

        self.assertTrue(check_causal_lm_output_quality(model_id, generated_tokens))

    @slow
    @pytest.mark.run_slow
    def test_llama_text_generation_with_custom_sdpa_and_kv_cache_8da4w_8we(self):
        model_id = "NousResearch/Llama-3.2-1B"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = ExecuTorchModelForCausalLM.from_pretrained(
            model_id,
            recipe="xnnpack",
            attn_implementation="custom_sdpa",
            use_custom_kv_cache=True,
            **{"qlinear": True, "qembeeding": True},
        )
        self.assertIsInstance(model, ExecuTorchModelForCausalLM)
        self.assertIsInstance(model.model, ExecuTorchModule)
        generated_text = model.text_generation(
            tokenizer=tokenizer,
            prompt="Simply put, the theory of relativity states that",
            max_seq_len=32,
        )
        logging.info(f"\nGenerated text:\n\t{generated_text}")
        generated_tokens = tokenizer(generated_text, return_tensors="pt").input_ids

        # Free memory before loading eager for quality check
        del model
        del tokenizer
        gc.collect()

        self.assertTrue(check_causal_lm_output_quality(model_id, generated_tokens))
