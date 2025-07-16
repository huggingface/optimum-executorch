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
import sys
import unittest

import pytest
from executorch.extension.pybindings.portable_lib import ExecuTorchModule
from transformers import AutoTokenizer
from transformers.testing_utils import slow

from optimum.executorch import ExecuTorchModelForCausalLM
from optimum.utils.import_utils import is_transformers_version

from ..utils import check_causal_lm_output_quality


os.environ["TOKENIZERS_PARALLELISM"] = "false"
is_ci = os.environ.get("GITHUB_ACTIONS") == "true"
is_linux_ci = sys.platform.startswith("linux") and is_ci


@pytest.mark.skipif(
    is_transformers_version("<", "4.53.1") or is_linux_ci,
    reason="Only available on transformers >= 4.53.1",
)
class ExecuTorchModelIntegrationTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @slow
    @pytest.mark.run_slow
    def test_smollm3_text_generation_with_custom_sdpa_and_kv_cache_8da4w_8we(self):
        model_id = "HuggingFaceTB/SmolLM3-3B"
        prompt = "Give me a brief explanation of gravity in simple terms."
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
            prompt=prompt,
            max_seq_len=64,
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
    @pytest.mark.portable
    @pytest.mark.skipif(is_ci, reason="Too big for CI runners")
    def test_smollm3_text_generation_portable(self):
        model_id = "HuggingFaceTB/SmolLM3-3B"
        prompt = "Give me a brief explanation of gravity in simple terms."
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = ExecuTorchModelForCausalLM.from_pretrained(model_id, recipe="portable")
        self.assertIsInstance(model, ExecuTorchModelForCausalLM)
        self.assertIsInstance(model.model, ExecuTorchModule)
        generated_text = model.text_generation(
            tokenizer=tokenizer,
            prompt=prompt,
            max_seq_len=64,
        )
        logging.info(f"\nGenerated text:\n\t{generated_text}")
        generated_tokens = tokenizer(generated_text, return_tensors="pt").input_ids

        # Free memory before loading eager for quality check
        del model
        del tokenizer
        gc.collect()

        self.assertTrue(check_causal_lm_output_quality(model_id, generated_tokens))
