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
import sys
import unittest

import pytest
from transformers import AutoProcessor, AutoTokenizer
from transformers.testing_utils import slow

from optimum.executorch import ExecuTorchModelForMultiModalToText

from ..utils import check_multimodal_output_quality


is_linux_ci = sys.platform.startswith("linux") and os.environ.get("GITHUB_ACTIONS") == "true"


os.environ["TOKENIZERS_PARALLELISM"] = "false"


class ExecuTorchModelIntegrationTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @slow
    @pytest.mark.run_slow
    @pytest.mark.skipif(is_linux_ci, reason="OOM")
    def test_smolvlm_with_custom_sdpa_kv_cache_8da4w_8we(self):
        model_id = "HuggingFaceTB/SmolVLM-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        processor = AutoProcessor.from_pretrained(model_id)
        conversation = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "url": "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg",
                    },
                    {"type": "text", "text": "Can you describe this image?"},
                ],
            },
        ]

        model = ExecuTorchModelForMultiModalToText.from_pretrained(
            # model_id,
            "/home/jackzhxng/models/smolvlm",
            recipe="xnnpack",
            task="multimodal-text-to-text",
            use_custom_sdpa=True,
            use_custom_kv_cache=True,
            qlinear="8da4w",
            qlinear_group_size=32,
            # Can't quantize the encoder a the moment, hidden dim of 4304 doesn't fit ExecuTorch's
            # XNNPack 32-group size quantized kernels. See https://github.com/pytorch/executorch/issues/14221.
            qembedding_config="8w",
        )

        # Generate
        generated_text = model.text_generation(
            processor=processor,
            tokenizer=tokenizer,
            input_conversation=conversation,
            max_seq_len=64,
        )
        logging.info(f"\nGenerated text:\n\t{generated_text}")
        generated_tokens = tokenizer(generated_text, return_tensors="pt").input_ids
        breakpoint()

        del model
        del tokenizer
        gc.collect()

        # Should be something like: 'Okay, let's analyze this image and discuss potential
        # cautions for visiting this location. Based on the picture, we're looking at a
        # serene lake scene with mountains in the background, a wooden pier extending into
        # the water, and a generally calm atmosphere.'
        self.assertTrue("Statue" in generated_text)
        self.assertTrue("Liberty" in generated_text)
        self.assertTrue(
            check_multimodal_output_quality(model_id, generated_tokens, conversation, max_perplexity_threshold=5)
        )
