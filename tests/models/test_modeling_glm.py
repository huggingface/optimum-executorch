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
import unittest

import pytest
import torchao
from executorch.extension.pybindings.portable_lib import ExecuTorchModule
from packaging.version import parse
from transformers import AutoTokenizer
from transformers.testing_utils import slow

from optimum.executorch import ExecuTorchModelForCausalLM

from ..utils import check_causal_lm_output_quality


os.environ["TOKENIZERS_PARALLELISM"] = "false"


class ExecuTorchModelIntegrationTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @slow
    @pytest.mark.run_slow
    @pytest.mark.skipif(
        parse(torchao.__version__) < parse("0.11.0"),
        reason="Quantization is only available on torchao >= 0.11.0.",
    )
    def test_glm_text_generation_with_custom_sdpa_and_kv_cache_8da4w_8we(self):
        model_id = "THUDM/glm-edge-1.5b-chat"
        prompt = "hello!"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = ExecuTorchModelForCausalLM.from_pretrained(
            model_id,
            recipe="xnnpack",
            attn_implementation="custom_sdpa",
            use_custom_kv_cache=True,
            **{"qlinear": "8da4w", "qembeeding": "8w"},
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
