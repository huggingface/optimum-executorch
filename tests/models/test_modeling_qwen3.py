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
from executorch import version
from executorch.extension.pybindings.portable_lib import ExecuTorchModule
from packaging.version import parse
from transformers import AutoTokenizer
from transformers.testing_utils import slow

from optimum.executorch import ExecuTorchModelForCausalLM

from ..utils import check_causal_lm_output_quality


os.environ["TOKENIZERS_PARALLELISM"] = "false"

is_linux_ci = sys.platform.startswith("linux") and os.environ.get("GITHUB_ACTIONS") == "true"


class ExecuTorchModelIntegrationTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @slow
    @pytest.mark.run_slow
    @pytest.mark.skipif(
        parse(torchao.__version__) < parse("0.11.0.dev0"),
        reason="Only available on torchao >= 0.11.0.dev0",
    )
    def test_qwen3_export_to_executorch(self):
        model_id = "Qwen/Qwen3-0.6B"
        task = "text-generation"
        recipe = "xnnpack"
        with tempfile.TemporaryDirectory() as tempdir:
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
            self.assertTrue(os.path.exists(f"{tempdir}/executorch/model.pte"))
            model = ExecuTorchModelForCausalLM.from_pretrained(f"{tempdir}/executorch")
            self._test_qwen3_text_generation(model_id, model)

    def _test_qwen3_text_generation(self, model_id: str, model: ExecuTorchModelForCausalLM):
        self.assertIsInstance(model, ExecuTorchModelForCausalLM)
        self.assertIsInstance(model.model, ExecuTorchModule)

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        generated_text = model.text_generation(
            tokenizer=tokenizer,
            prompt="Give me a short introduction to large language model.",
            max_seq_len=64,
        )
        logging.info(f"\nGenerated text:\n\t{generated_text}")
        generated_tokens = tokenizer(generated_text, return_tensors="pt").input_ids

        # Free memory before loading eager for quality check
        del model
        del tokenizer
        gc.collect()

        self.assertTrue(check_causal_lm_output_quality(model_id, generated_tokens))

    def _helper_qwen3_text_generation(self, recipe: str):
        model_id = "Qwen/Qwen3-0.6B"
        model = ExecuTorchModelForCausalLM.from_pretrained(model_id, recipe=recipe)
        self._test_qwen3_text_generation(model_id, model)

    @slow
    @pytest.mark.run_slow
    @pytest.mark.skipif(is_linux_ci, reason="OOM on linux runner")
    def test_qwen3_text_generation(self):
        self._helper_qwen3_text_generation(recipe="xnnpack")

    @slow
    @pytest.mark.run_slow
    @pytest.mark.portable
    @pytest.mark.skipif(is_linux_ci, reason="OOM on linux runner")
    @pytest.mark.skipif(
        parse(version.__version__) < parse("0.7.0"),
        reason="Fixed on executorch >= 0.7.0",
    )
    def test_qwen3_text_generation_portable(self):
        self._helper_qwen3_text_generation(recipe="portable")

    @slow
    @pytest.mark.run_slow
    @pytest.mark.skipif(is_linux_ci, reason="OOM on linux runner")
    def test_qwen3_text_generation_with_custom_sdpa(self):
        model_id = "Qwen/Qwen3-0.6B"
        prompt = "Give me a short introduction to large language model."
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
    def test_qwen3_text_generation_with_custom_sdpa_float16(self):
        model_id = "Qwen/Qwen3-0.6B"
        prompt = "Give me a short introduction to large language model."
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        # ExecuTorch model + custom sdpa
        kwargs = {"dtype": "float16"}
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
        parse(torchao.__version__) < parse("0.11.0.dev0"),
        reason="Only available on torchao >= 0.11.0.dev0",
    )
    def test_qwen3_text_generation_with_custom_sdpa_8da4w_8we(self):
        model_id = "Qwen/Qwen3-0.6B"
        prompt = "Give me a short introduction to large language model."
        tokenizer = AutoTokenizer.from_pretrained(model_id)

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
        generated_text = model.text_generation(
            tokenizer=tokenizer,
            prompt=prompt,
            max_seq_len=128,
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
        parse(transformers.__version__) < parse("4.52.0.dev0"),
        reason="Only available on transformers >= 4.52.0.dev0",
    )
    def test_qwen3_text_generation_with_custom_sdpa_and_kv_cache(self):
        model_id = "Qwen/Qwen3-0.6B"
        prompt = "Give me a short introduction to large language model."
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        # ExecuTorch model + custom sdpa
        model = ExecuTorchModelForCausalLM.from_pretrained(
            model_id,
            recipe="xnnpack",
            attn_implementation="custom_sdpa",
            use_custom_kv_cache=True,
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
        parse(transformers.__version__) < parse("4.52.0") or parse(torchao.__version__) < parse("0.11.0"),
        reason="Only available on transformers >= 4.52.0 and torchao >= 0.11.0",
    )
    def test_qwen3_text_generation_with_custom_sdpa_and_kv_cache_8da4w_8we(self):
        model_id = "Qwen/Qwen3-0.6B"
        prompt = "Give me a short introduction to large language model."
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = ExecuTorchModelForCausalLM.from_pretrained(
            model_id,
            recipe="xnnpack",
            attn_implementation="custom_sdpa",
            use_custom_kv_cache=True,
            **{"qlinear": "8da4w", "qembedding": "8w"},
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
