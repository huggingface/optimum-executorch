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
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from executorch.extension.pybindings.portable_lib import ExecuTorchModule
from huggingface_hub import HfApi
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
)

from optimum.executorch import ExecuTorchModelForCausalLM
from optimum.executorch.modeling import _FILE_PATTERN
from optimum.exporters.executorch import main_export
from optimum.utils.file_utils import find_files_matching_pattern

from ..utils import check_causal_lm_output_quality


os.environ["TOKENIZERS_PARALLELISM"] = "false"


class ExecuTorchModelIntegrationTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def test_export_cli_helps_no_raise(self):
        subprocess.run(
            "optimum-cli export executorch --help",
            shell=True,
            check=True,
        )

    def test_load_cached_model_from_hub(self):
        model_id = "optimum-internal-testing/tiny-random-llama"

        model = ExecuTorchModelForCausalLM.from_pretrained(model_id, recipe="xnnpack")
        self.assertIsInstance(model, ExecuTorchModelForCausalLM)
        self.assertTrue(hasattr(model, "model"))
        self.assertIsInstance(model.model, ExecuTorchModule)

    def test_load_et_model_from_hub(self):
        model_id = "optimum-internal-testing/tiny-random-llama"

        model = ExecuTorchModelForCausalLM.from_pretrained(model_id, revision="executorch")
        self.assertIsInstance(model, ExecuTorchModelForCausalLM)
        self.assertTrue(hasattr(model, "model"))
        self.assertIsInstance(model.model, ExecuTorchModule)

        model = ExecuTorchModelForCausalLM.from_pretrained(model_id, revision="executorch-subfolder")
        self.assertIsInstance(model, ExecuTorchModelForCausalLM)
        self.assertTrue(hasattr(model, "model"))
        self.assertIsInstance(model.model, ExecuTorchModule)

    def test_load_cached_model_from_local_path(self):
        model_id = "optimum-internal-testing/tiny-random-llama"
        recipe = "xnnpack"

        with tempfile.TemporaryDirectory() as tempdir:
            # Export to a local dir
            main_export(
                model_name_or_path=model_id,
                recipe=recipe,
                output_dir=tempdir,
                task="text-generation",
            )
            self.assertTrue(os.path.exists(f"{tempdir}/model.pte"))

            # Load the exported model from a local dir
            model = ExecuTorchModelForCausalLM.from_pretrained(tempdir)
            self.assertIsInstance(model, ExecuTorchModelForCausalLM)
            self.assertTrue(hasattr(model, "model"))
            self.assertIsInstance(model.model, ExecuTorchModule)

    def test_find_files_matching_pattern(self):
        model_id = "optimum-internal-testing/tiny-random-llama"

        # hub model
        for revision in ("main", "executorch"):
            pte_files = find_files_matching_pattern(model_id, pattern=_FILE_PATTERN, revision=revision)
            self.assertTrue(len(pte_files) == 0 if revision == "main" else len(pte_files) > 0)

        # local model
        api = HfApi()
        with TemporaryDirectory() as tmpdirname:
            for revision in ("main", "executorch"):
                local_dir = Path(tmpdirname) / revision
                api.snapshot_download(repo_id=model_id, local_dir=local_dir, revision=revision)
                pte_files = find_files_matching_pattern(local_dir, pattern=_FILE_PATTERN, revision=revision)
                self.assertTrue(len(pte_files) == 0 if revision == "main" else len(pte_files) > 0)

    def test_export_with_custom_sdpa(self):
        model_id = "optimum-internal-testing/tiny-random-llama"
        with tempfile.TemporaryDirectory() as tempdir:
            subprocess.run(
                f"optimum-cli export executorch \
                    --model {model_id} \
                    --task 'text-generation' \
                    --recipe 'xnnpack' \
                    --output_dir {tempdir}/executorch",
                shell=True,
                check=True,
            )
            self.assertTrue(os.path.exists(f"{tempdir}/executorch/model.pte"))

    def test_eager_text_generation_with_custom_sdpa(self):
        model_id = "HuggingFaceTB/SmolLM2-135M"
        prompt = "My favourite condiment is "
        max_seq_len = 32
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        # Eager model + custom sdpa
        cache_implementation = "static"
        attn_implementation = "custom_sdpa"
        eager_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            attn_implementation=attn_implementation,
            generation_config=GenerationConfig(
                use_cache=True,
                cache_implementation=cache_implementation,
                max_length=max_seq_len,
                cache_config={
                    "batch_size": 1,
                    "max_cache_len": max_seq_len,
                },
            ),
        )
        self.assertTrue(eager_model.config._attn_implementation, attn_implementation)
        eager_inputs = tokenizer(prompt, return_tensors="pt").to(eager_model.device)
        eager_generated_ids = eager_model.generate(**eager_inputs, max_new_tokens=max_seq_len, temperature=0)
        eager_generated_text = tokenizer.batch_decode(eager_generated_ids, skip_special_tokens=True)[0]
        logging.info(f"\nEager generated text:\n\t{eager_generated_text}")
        self.assertTrue(check_causal_lm_output_quality(model_id, eager_generated_ids))

    def test_removing_padding_idx_embedding_pass(self):
        class ModuleWithEmbedding(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.emb = torch.nn.Embedding(10, 3, padding_idx=0)

            def forward(self, x):
                return self.emb(x) + torch.ops.aten.embedding.default(self.emb.weight, x, padding_idx=1)

        test_model = ModuleWithEmbedding()
        example_inputs = (torch.tensor([1, 2, 4, 5, 4, 3, 2, 9]),)
        exported_model = torch.export.export(test_model, example_inputs)

        from executorch.exir import to_edge_transform_and_lower
        from executorch.exir.dialects._ops import ops as exir_ops

        from optimum.executorch.passes.remove_padding_idx_embedding_pass import RemovePaddingIdxEmbeddingPass

        et_model = to_edge_transform_and_lower(
            exported_model,
            transform_passes=[RemovePaddingIdxEmbeddingPass()],
        )
        self.assertTrue(
            all(
                len(node.args) < 3
                for node in et_model.exported_program().graph_module.graph.nodes
                if node.op == "call_function" and node.target == exir_ops.edge.aten.embedding.default
            )
        )
