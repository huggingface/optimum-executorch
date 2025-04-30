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
import sys
import tempfile
import unittest

import pytest
from executorch.extension.pybindings.portable_lib import ExecuTorchModule
from transformers import AutoTokenizer
from transformers.testing_utils import slow

from optimum.executorch import ExecuTorchModelForCausalLM

from ..utils import check_causal_lm_output_quality


is_linux_ci = sys.platform.startswith("linux") and os.environ.get("GITHUB_ACTIONS") == "true"


@pytest.mark.skipif(is_linux_ci, reason="OOM on linux runner")
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
                f"optimum-cli export executorch --model {model_id} --task {task} --recipe {recipe} --output_dir {out_dir}",
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
    def test_gemma_text_generation(self):
        # TODO: Switch to use google/gemma-2b once https://github.com/huggingface/optimum/issues/2127 is fixed
        # model_id = "google/gemma-2b"
        model_id = "weqweasdas/RM-Gemma-2B"
        model = ExecuTorchModelForCausalLM.from_pretrained(
            model_id,
            recipe="xnnpack",
            **{"dtype": "float16"},
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
        generated_tokens = tokenizer(generated_text, return_tensors="pt").input_ids

        # Free memory before loading eager for quality check
        del model
        del tokenizer
        gc.collect()

        self.assertTrue(check_causal_lm_output_quality(model_id, generated_tokens))
