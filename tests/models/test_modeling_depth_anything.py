# coding=utf-8
# Copyright 2025 The HuggingFace Team. All rights reserved.
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
from parameterized import parameterized
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from transformers.testing_utils import slow

from optimum.executorch import ExecuTorchModelForDepthEstimation
from optimum.exporters.executorch import main_export

from ..utils import check_close_recursively


class ExecuTorchModelIntegrationTest(unittest.TestCase):
    DEPTH_ANYTHING_MODEL_NAMES = [
        "depth-anything/Depth-Anything-V2-Small-hf",
    ]

    @slow
    @pytest.mark.run_slow
    def test_depth_anything_export_to_executorch(self):
        """Test CLI export of Depth-Anything model"""
        model_id = "depth-anything/Depth-Anything-V2-Small-hf"
        task = "depth-estimation"
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
    @parameterized.expand(
        DEPTH_ANYTHING_MODEL_NAMES,
        skip_on_empty=True,
    )
    def test_depth_anything_export_and_inference(self, model_name):
        with tempfile.TemporaryDirectory() as tempdir:
            # Export the model to ExecuTorch
            main_export(
                model_name_or_path=model_name,
                output_dir=tempdir,
                recipe="xnnpack",
                quantize=False,
            )

            # Load the exported model
            executorch_model = ExecuTorchModelForDepthEstimation.from_pretrained(tempdir)

            # Load the original model for comparison
            original_model = AutoModelForDepthEstimation.from_pretrained(model_name).eval()

            # Prepare test input
            image_processor = AutoImageProcessor.from_pretrained(model_name)
            
            # Create a dummy RGB image
            test_image = Image.new("RGB", (640, 480), color="red")
            pixel_values = image_processor(images=test_image, return_tensors="pt").pixel_values

            # Run inference with both models
            with torch.no_grad():
                original_output = original_model(pixel_values).predicted_depth
                executorch_output = executorch_model(pixel_values)

            # Check output shape matches
            self.assertEqual(
                original_output.shape,
                executorch_output.shape,
                f"Output shapes don't match: {original_output.shape} vs {executorch_output.shape}"
            )

            # Check that outputs are reasonably close (allowing for quantization differences)
            # We use a relatively high tolerance since depth estimation can have some variation
            torch.testing.assert_close(
                original_output,
                executorch_output,
                rtol=1e-2,
                atol=1e-2,
                msg=f"Outputs differ significantly for {model_name}"
            )

    @slow
    @pytest.mark.run_slow
    @pytest.mark.portable
    @parameterized.expand(
        DEPTH_ANYTHING_MODEL_NAMES,
        skip_on_empty=True,
    )
    def test_depth_anything_portable_export(self, model_name):
        """Test exporting with portable recipe (CPU-only, no backend delegation)"""
        with tempfile.TemporaryDirectory() as tempdir:
            # Export with portable recipe
            main_export(
                model_name_or_path=model_name,
                output_dir=tempdir,
                recipe="portable",
                quantize=False,
            )

            # Load and verify the model can be instantiated
            executorch_model = ExecuTorchModelForDepthEstimation.from_pretrained(tempdir)
            
            # Prepare test input
            image_processor = AutoImageProcessor.from_pretrained(model_name)
            test_image = Image.new("RGB", (640, 480), color="blue")
            pixel_values = image_processor(images=test_image, return_tensors="pt").pixel_values

            # Run inference
            with torch.no_grad():
                output = executorch_model(pixel_values)
            
            # Basic sanity checks
            self.assertIsInstance(output, torch.Tensor)
            self.assertEqual(len(output.shape), 3)  # [batch, height, width]
            self.assertEqual(output.shape[0], 1)    # batch size should be 1

    @slow
    @pytest.mark.run_slow
    @pytest.mark.quantization
    @parameterized.expand(
        DEPTH_ANYTHING_MODEL_NAMES,
        skip_on_empty=True,
    )
    def test_depth_anything_with_quantization(self, model_name):
        """Test exporting with quantization enabled"""
        with tempfile.TemporaryDirectory() as tempdir:
            # Export with quantization
            main_export(
                model_name_or_path=model_name,
                output_dir=tempdir,
                recipe="xnnpack",
                quantize=True,
            )

            # Load the quantized model
            executorch_model = ExecuTorchModelForDepthEstimation.from_pretrained(tempdir)
            
            # Prepare test input
            image_processor = AutoImageProcessor.from_pretrained(model_name)
            test_image = Image.new("RGB", (640, 480), color="green")
            pixel_values = image_processor(images=test_image, return_tensors="pt").pixel_values

            # Run inference
            with torch.no_grad():
                output = executorch_model(pixel_values)
            
            # Basic sanity checks for quantized model
            self.assertIsInstance(output, torch.Tensor)
            self.assertEqual(len(output.shape), 3)  # [batch, height, width]
            self.assertEqual(output.shape[0], 1)    # batch size should be 1


if __name__ == "__main__":
    unittest.main()