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
from executorch import version
from executorch.extension.pybindings.portable_lib import ExecuTorchModule
from transformers import AutoConfig, AutoModelForImageClassification
from transformers.testing_utils import slow
import coremltools as ct
from optimum.executorch import ExecuTorchModelForImageClassification

from ..utils import check_close_recursively


class ExecuTorchModelIntegrationTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @slow
    @pytest.mark.run_slow
    def test_efficientnet_export_to_executorch(self):
        model_id = "google/efficientnet-b7"  # ~66M params
        task = "image-classification"
        recipe = "xnnpack"
        with tempfile.TemporaryDirectory() as tempdir:
            subprocess.run(
                f"optimum-cli export executorch --model {model_id} --task {task} --recipe {recipe} --output_dir {tempdir}/executorch",
                shell=True,
                check=True,
            )
            self.assertTrue(os.path.exists(f"{tempdir}/executorch/model.pte"))

    @slow
    @pytest.mark.run_slow
    @pytest.mark.skipif(
        version.__version__ < "0.6.0",
        reason="The fix in XNNPACK is cherry-picked in 0.6.0 release",
    )
    def test_efficientnet_image_classification(self):
        model_id = "google/efficientnet-b0"  # ~5.3M params

        config = AutoConfig.from_pretrained(model_id)
        batch_size = 1
        num_channels = config.num_channels
        height = config.image_size
        width = config.image_size
        pixel_values = torch.rand(batch_size, num_channels, height, width)

        # Test fetching and lowering the model to ExecuTorch
        et_model = ExecuTorchModelForImageClassification.from_pretrained(model_id=model_id, recipe="xnnpack")
        self.assertIsInstance(et_model, ExecuTorchModelForImageClassification)
        self.assertIsInstance(et_model.model, ExecuTorchModule)

        eager_model = AutoModelForImageClassification.from_pretrained(model_id).eval().to("cpu")
        with torch.no_grad():
            eager_output = eager_model(pixel_values)
            et_output = et_model.forward(pixel_values)

        # Compare with eager outputs
        self.assertTrue(check_close_recursively(eager_output.logits, et_output))
    
    @slow
    @pytest.mark.run_slow
    def test_efficientnet_image_classification_coreml(self):
        model_id = "google/efficientnet-b0"  # ~5.3M params

        config = AutoConfig.from_pretrained(model_id)
        batch_size = 1
        num_channels = config.num_channels
        height = config.image_size
        width = config.image_size
        pixel_values = torch.rand(batch_size, num_channels, height, width)

        # Test fetching and lowering the model to ExecuTorch
        et_model = ExecuTorchModelForImageClassification.from_pretrained(model_id=model_id, recipe="coreml", kwargs={"recipe_kwargs": {"compute_precision": ct.PRECISION.FLOAT32}})
        self.assertIsInstance(et_model, ExecuTorchModelForImageClassification)
        self.assertIsInstance(et_model.model, ExecuTorchModule)

        eager_model = AutoModelForImageClassification.from_pretrained(model_id).eval().to("cpu")
        with torch.no_grad():
            eager_output = eager_model(pixel_values)
            et_output = et_model.forward(pixel_values)

        # Compare with eager outputs
        self.assertTrue(check_close_recursively(eager_output.logits, et_output))
