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

import os
import tempfile
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from executorch.extension.pybindings.portable_lib import ExecuTorchModule
from huggingface_hub import HfApi

from optimum.executorch import ExecuTorchModelForCausalLM
from optimum.exporters.executorch import main_export
from optimum.utils.file_utils import _find_files_matching_pattern


class ExecuTorchModelIntegrationTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def test_load_model_from_hub(self):
        model_id = "optimum-internal-testing/tiny-random-llama"

        model = ExecuTorchModelForCausalLM.from_pretrained(model_id, recipe="xnnpack")
        self.assertIsInstance(model, ExecuTorchModelForCausalLM)
        self.assertIsInstance(model.model, ExecuTorchModule)

    def test_load_et_model_from_hub(self):
        model_id = "optimum-internal-testing/tiny-random-llama"

        model = ExecuTorchModelForCausalLM.from_pretrained(
            model_id, export=False, revision="executorch", recipe="xnnpack"
        )
        self.assertIsInstance(model, ExecuTorchModelForCausalLM)
        self.assertIsInstance(model.model, ExecuTorchModule)

    def test_load_model_from_local_path(self):
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
            self.assertIsInstance(model.model, ExecuTorchModule)

    def test_find_files_matching_pattern(self):
        model_id = "optimum-internal-testing/tiny-random-llama"
        pattern = r"(.*).pte$"

        # hub model
        for revision in ("main", "executorch"):
            pte_files = _find_files_matching_pattern(model_id, pattern=pattern, revision=revision)
            self.assertTrue(len(pte_files) == 0 if revision == "main" else len(pte_files) > 0)

        # local model
        api = HfApi()
        with TemporaryDirectory() as tmpdirname:
            for revision in ("main", "executorch"):
                local_dir = Path(tmpdirname) / revision
                api.snapshot_download(repo_id=model_id, local_dir=local_dir, revision=revision)
                pte_files = _find_files_matching_pattern(
                    local_dir, pattern=pattern, revision=revision, subfolder=revision
                )
                self.assertTrue(len(pte_files) == 0 if revision == "main" else len(pte_files) > 0)
