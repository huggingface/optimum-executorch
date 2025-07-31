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

import gc
import logging
import os
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import pytest
import torch
from transformers.testing_utils import slow

from optimum.executorch import ExecuTorchModelForCausalLM
from optimum.exporters.executorch.integrations import ImageTextToTextExportableModule
from optimum.utils.import_utils import is_transformers_version


is_linux_ci = sys.platform.startswith("linux") and os.environ.get("GITHUB_ACTIONS") == "true"


os.environ["TOKENIZERS_PARALLELISM"] = "false"


@pytest.mark.skipif(
    is_transformers_version("<", "4.52.0.dev0"),
    reason="Only available on transformers >= 4.52.0.dev0",
)
class ImageTextToTextExportTest(unittest.TestCase):
    def setUp(self):
        # Mock multimodal model configuration
        self.mock_config = Mock()
        self.mock_config.text_config = Mock()
        self.mock_config.text_config.use_cache = True
        self.mock_config.text_config.hidden_size = 768
        self.mock_config.text_config.num_hidden_layers = 12
        self.mock_config.vision_config = Mock()
        self.mock_config.vision_config.image_size = 224

        # Mock generation config
        self.mock_generation_config = Mock()
        self.mock_generation_config.use_cache = True
        self.mock_generation_config.cache_implementation = "static"
        self.mock_generation_config.max_length = 2048
        self.mock_generation_config.cache_config = {
            "batch_size": 1,
            "max_cache_len": 2048,
        }

        # Mock model
        self.mock_model = Mock()
        self.mock_model.config = self.mock_config
        self.mock_model.generation_config = self.mock_generation_config
        self.mock_model.device = torch.device("cpu")
        self.mock_model.dtype = torch.float32

        # Mock language model and vision tower
        self.mock_model.language_model = Mock()
        self.mock_model.vision_tower = Mock()

    def test_image_text_to_text_module_initialization(self):
        """Test that ImageTextToTextExportableModule initializes correctly"""
        with patch("optimum.exporters.executorch.integrations.save_config_to_constant_methods") as mock_save:
            mock_save.return_value = {"get_max_seq_len": 2048}
            
            module = ImageTextToTextExportableModule(self.mock_model)
            
            self.assertEqual(module.model, self.mock_model)
            self.assertEqual(module.config, self.mock_config)
            self.assertFalse(module.use_custom_kv_cache)
            self.assertFalse(module.use_custom_sdpa)
            mock_save.assert_called_once_with(self.mock_config.text_config, self.mock_generation_config)

    def test_vision_embedding_export_inputs_preparation(self):
        """Test vision embedding export inputs preparation"""
        with patch("optimum.exporters.executorch.integrations.save_config_to_constant_methods") as mock_save:
            mock_save.return_value = {"get_max_seq_len": 2048}
            
            module = ImageTextToTextExportableModule(self.mock_model)
            pixel_values, dynamic_shapes, strict = module._prepare_vision_embedding_export_inputs()
            
            self.assertEqual(pixel_values.shape, (1, 3, 224, 224))  # batch, channels, height, width
            self.assertIsNone(dynamic_shapes)
            self.assertFalse(strict)

    def test_text_embedding_export_inputs_preparation(self):
        """Test text embedding export inputs preparation"""
        with patch("optimum.exporters.executorch.integrations.save_config_to_constant_methods") as mock_save:
            mock_save.return_value = {"get_max_seq_len": 2048, "sliding_window": float("inf")}
            
            module = ImageTextToTextExportableModule(self.mock_model)
            inputs_embeds, cache_position, dynamic_shapes, strict = module._prepare_text_embedding_export_inputs()
            
            self.assertEqual(inputs_embeds.shape, (1, 3, 768))  # batch, seq_len, hidden_size
            self.assertEqual(cache_position.shape, (3,))  # seq_len
            self.assertIn("inputs_embeds", dynamic_shapes)
            self.assertIn("cache_position", dynamic_shapes)
            self.assertFalse(strict)

    def test_export_method_structure(self):
        """Test that export method has correct structure"""
        with patch("optimum.exporters.executorch.integrations.save_config_to_constant_methods") as mock_save:
            with patch("optimum.exporters.executorch.integrations.VisionEncoderExportableModule") as mock_vision:
                with patch("optimum.exporters.executorch.integrations.is_transformers_version") as mock_version:
                    mock_save.return_value = {"get_max_seq_len": 2048, "sliding_window": float("inf")}
                    mock_version.return_value = True
                    
                    # Mock vision encoder export
                    mock_vision_instance = Mock()
                    mock_vision_instance.export.return_value = {"model": Mock()}
                    mock_vision.return_value = mock_vision_instance
                    
                    # Mock transformers module
                    with patch("transformers.integrations.executorch.TorchExportableModuleForImageTextLM") as mock_text_module:
                        mock_text_instance = Mock()
                        mock_text_instance.export.return_value = Mock()
                        mock_text_module.return_value = mock_text_instance
                        
                        module = ImageTextToTextExportableModule(self.mock_model)
                        result = module.export()
                        
                        # Verify structure
                        self.assertIn("vision_encoder", result)
                        self.assertIn("text_decoder", result)
                        
                        # Verify calls
                        mock_vision.assert_called_once_with(self.mock_model)
                        mock_text_module.assert_called_once()

    def test_validation_errors(self):
        """Test validation errors for invalid configurations"""
        # Test missing text_config
        bad_config = Mock()
        bad_config.vision_config = Mock()
        # Missing text_config
        
        bad_model = Mock()
        bad_model.config = bad_config
        
        with patch("optimum.exporters.executorch.tasks.image_text_to_text.AutoConfig") as mock_auto_config:
            mock_auto_config.from_pretrained.return_value = bad_config
            
            from optimum.exporters.executorch.tasks.image_text_to_text import load_image_text_to_text_model
            
            with self.assertRaises(ValueError) as context:
                load_image_text_to_text_model("test_model")
            
            self.assertIn("text_config", str(context.exception))

    @slow
    @pytest.mark.run_slow
    @pytest.mark.skipif(is_linux_ci, reason="OOM on linux runner")
    def test_cli_export_integration(self):
        """Test CLI integration for image-text-to-text task"""
        # This would test the actual CLI command but requires a real model
        # For now, just test that the task is registered correctly
        from optimum.exporters.executorch.task_registry import task_registry
        
        # Discover tasks to populate registry
        from optimum.exporters.executorch.task_registry import discover_tasks
        discover_tasks()
        
        self.assertIn("image-text-to-text", task_registry)

    def test_transformers_version_requirement(self):
        """Test that export requires proper transformers version"""
        with patch("optimum.exporters.executorch.integrations.save_config_to_constant_methods") as mock_save:
            with patch("optimum.exporters.executorch.integrations.is_transformers_version") as mock_version:
                mock_save.return_value = {"get_max_seq_len": 2048}
                mock_version.return_value = False  # Simulate old transformers version
                
                module = ImageTextToTextExportableModule(self.mock_model)
                
                with self.assertRaises(ValueError) as context:
                    module.export()
                
                self.assertIn("transformers > 4.52.0", str(context.exception))


if __name__ == "__main__":
    unittest.main()