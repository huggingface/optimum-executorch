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
import subprocess
import tempfile
import unittest

import pytest
import torch
from executorch.extension.pybindings.portable_lib import ExecuTorchModule

from optimum.executorch import ExecuTorchModelForImageClassification
from transformers import AutoConfig, AutoModelForImageClassification
from transformers.testing_utils import slow

from ..utils import check_close_recursively


class ExecuTorchModelIntegrationTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @slow
    @pytest.mark.run_slow
    def test_deit_export_to_executorch(self):
        model_id = "facebook/deit-base-distilled-patch16-224"
        task = "image-classification"
        recipe = "xnnpack"
        with tempfile.TemporaryDirectory() as tempdir:
            subprocess.run(
                f"optimum-cli export executorch --model {model_id} --task {task} --recipe {recipe} --output_dir {tempdir}/executorch",
                shell=True,
                check=True,
            )
            self.assertTrue(os.path.exists(f"{tempdir}/executorch/model.pte"))

    def _helper_deit_image_classification(self, recipe: str):
        model_id = "facebook/deit-base-distilled-patch16-224"

        config = AutoConfig.from_pretrained(model_id)
        batch_size = 1
        num_channels = config.num_channels
        height = config.image_size
        width = config.image_size
        pixel_values = torch.rand(batch_size, num_channels, height, width)

        # Test fetching and lowering the model to ExecuTorch
        et_model = ExecuTorchModelForImageClassification.from_pretrained(
            model_id=model_id, recipe=recipe
        )
        self.assertIsInstance(et_model, ExecuTorchModelForImageClassification)
        self.assertIsInstance(et_model.model, ExecuTorchModule)

        eager_model = (
            AutoModelForImageClassification.from_pretrained(model_id).eval().to("cpu")
        )
        with torch.no_grad():
            eager_output = eager_model(pixel_values)
            et_output = et_model.forward(pixel_values)

        # Compare with eager outputs
        self.assertTrue(check_close_recursively(eager_output.logits, et_output))

    @slow
    @pytest.mark.run_slow
    def test_deit_image_classification(self):
        self._helper_deit_image_classification(recipe="xnnpack")

    @slow
    @pytest.mark.run_slow
    @pytest.mark.portable
    def test_deit_image_classification_portable(self):
        self._helper_deit_image_classification(recipe="portable")

    @slow
    @pytest.mark.run_slow
    def test_deit_image_classification_qnn(self):
        self._helper_deit_image_classification(recipe="qnn_fp16_SM8650")
