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
import sys
import tempfile
import unittest

import pytest
import torch
from transformers.testing_utils import slow

from optimum.executorch import ExecuTorchModelForImageClassification

from ..utils import check_close_recursively


is_not_macos = sys.platform != "darwin"


class ExecuTorchModelIntegrationTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @slow
    @pytest.mark.run_slow
    def test_vit_export_to_executorch(self):
        model_id = "microsoft/resnet-50"
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
    @pytest.mark.skipif(is_not_macos, reason="Only runs on MacOS")
    def test_vit_image_classification_coreml_fp32_cpu(self):
        model_id = "microsoft/resnet-50"

        batch_size = 1
        num_channels = 3
        height = 224
        width = 224
        pixel_values = torch.rand(batch_size, num_channels, height, width)

        # Test fetching and lowering the model to ExecuTorch
        import coremltools as ct

        et_model = ExecuTorchModelForImageClassification.from_pretrained(
            model_id=model_id,
            recipe="coreml",
            recipe_kwargs={"compute_precision": ct.precision.FLOAT32, "compute_units": ct.ComputeUnit.CPU_ONLY},
        )
        et_output = et_model.forward(pixel_values)

        # Reference (using XNNPACK as reference because eager model currently segfaults in a PyTorch kernel)
        et_xnnpack = ExecuTorchModelForImageClassification.from_pretrained(
            model_id=model_id,
            recipe="xnnpack",
        )
        et_xnnpack_output = et_xnnpack.forward(pixel_values)

        # Compare with reference
        self.assertTrue(check_close_recursively(et_output, et_xnnpack_output))
