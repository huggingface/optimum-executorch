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
import torchao
import transformers
from executorch.extension.pybindings.portable_lib import ExecuTorchModule
from packaging.version import parse
from transformers import AutoTokenizer
from transformers.testing_utils import slow

from optimum.executorch import ExecuTorchModelForCausalLM
from optimum.utils.import_utils import is_transformers_version

from ..utils import check_causal_lm_output_quality


is_linux_ci = sys.platform.startswith("linux") and os.environ.get("GITHUB_ACTIONS") == "true"


os.environ["TOKENIZERS_PARALLELISM"] = "false"


@pytest.mark.skipif(
    is_transformers_version("<", "4.52.0.dev0"),
    reason="Only available on transformers >= 4.52.0.dev0",
)
class ExecuTorchModelIntegrationTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @slow
    @pytest.mark.run_slow
    def test_gemma3_export_to_executorch(self):
        # TODO: Until https://github.com/huggingface/optimum/issues/2127 is fixed, have to use non-gated model on CI
        # model_id = "google/gemma-3-1b-it"
        model_id = "unsloth/gemma-3-1b-it"
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
                    --qlinear \
                    --qembedding",
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
    @pytest.mark.skipif(is_linux_ci, reason="OOM on linux runner")
    def test_gemma3_text_generation(self):
        # TODO: Until https://github.com/huggingface/optimum/issues/2127 is fixed, have to use non-gated model on CI
        # model_id = "google/gemma-3-1b-it"
        model_id = "unsloth/gemma-3-1b-it"
        model = ExecuTorchModelForCausalLM.from_pretrained(
            model_id,
            recipe="xnnpack",
        )
        self.assertIsInstance(model, ExecuTorchModelForCausalLM)
        self.assertIsInstance(model.model, ExecuTorchModule)

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        generated_text = model.text_generation(
            tokenizer=tokenizer,
            prompt="Write a poem about a machine learning.",
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
    @pytest.mark.skipif(is_linux_ci, reason="OOM on linux runner")
    @pytest.mark.skipif(
        parse(transformers.__version__) < parse("4.53.0.dev0") or parse(torchao.__version__) < parse("0.11.0"),
        reason="Only available on transformers >= 4.53.0.dev0 and torchao >= 0.11.0",
    )
    def test_gemma3_text_generation_with_custom_sdpa(self):
        # TODO: Until https://github.com/huggingface/optimum/issues/2127 is fixed, have to use non-gated model on CI
        # model_id = "google/gemma-3-1b-it"
        model_id = "unsloth/gemma-3-1b-it"
        prompt = "Write a poem about a machine learning."
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        # ExecuTorch model + custom sdpa
        model = ExecuTorchModelForCausalLM.from_pretrained(
            model_id,
            recipe="xnnpack",
            attn_implementation="custom_sdpa",
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
    def test_gemma3_text_generation_with_custom_sdpa_float16(self):
        # TODO: Until https://github.com/huggingface/optimum/issues/2127 is fixed, have to use non-gated model on CI
        # model_id = "google/gemma-3-1b-it"
        model_id = "unsloth/gemma-3-1b-it"
        prompt = "Write a poem about a machine learning."
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        kwargs = {"dtype": "float16"}

        # ExecuTorch model + custom sdpa + float16
        model = ExecuTorchModelForCausalLM.from_pretrained(
            model_id,
            recipe="xnnpack",
            attn_implementation="custom_sdpa",
            **kwargs,
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
    @pytest.mark.skipif(
        parse(transformers.__version__) < parse("4.53.0.dev0") or parse(torchao.__version__) < parse("0.11.0"),
        reason="Only available on transformers >= 4.53.0.dev0 and torchao >= 0.11.0",
    )
    def test_gemma3_text_generation_with_custom_sdpa_8da4w_8we(self):
        # TODO: Until https://github.com/huggingface/optimum/issues/2127 is fixed, have to use non-gated model on CI
        # model_id = "google/gemma-3-1b-it"
        model_id = "unsloth/gemma-3-1b-it"
        prompt = "Write a poem about a machine learning."

        # ExecuTorch model + custom sdpa + 8da4w linear quantization + int8 embedding quantization
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
    @pytest.mark.skipif(
        parse(transformers.__version__) < parse("4.53.0.dev0") or parse(torchao.__version__) < parse("0.11.0"),
        reason="Only available on transformers >= 4.53.0.dev0 and torchao >= 0.11.0",
    )
    def test_gemma3_text_generation_with_custom_sdpa_kv_cache_8da4w_8we(self):
        # TODO: Until https://github.com/huggingface/optimum/issues/2127 is fixed, have to use non-gated model on CI
        # model_id = "google/gemma-3-1b-it"
        model_id = "unsloth/gemma-3-1b-it"
        prompt = "Write a poem about a machine learning."

        # ExecuTorch model + custom sdpa + 8da4w linear quantization + int8 embedding quantization
        kwargs = {"qlinear": True, "qembedding": True}
        model = ExecuTorchModelForCausalLM.from_pretrained(
            model_id,
            recipe="xnnpack",
            attn_implementation="custom_sdpa",
            use_custom_kv_cache=True,
            **kwargs,
        )
        self.assertIsInstance(model, ExecuTorchModelForCausalLM)
        self.assertIsInstance(model.model, ExecuTorchModule)

        tokenizer = AutoTokenizer.from_pretrained(model_id)
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
