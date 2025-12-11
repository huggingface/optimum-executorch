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
import unittest

import pytest
from transformers import MistralCommonBackend
from transformers.testing_utils import slow

from optimum.executorch import ExecuTorchModelForMultiModalToText

from ..utils import check_causal_lm_output_quality


class ExecuTorchModelIntegrationTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @slow
    @pytest.mark.run_slow
    def test_ministral_3_text_generation_with_custom_sdpa_and_kv_cache_8da4w_8we(self):
        model_id = "mistralai/Ministral-3-3B-Instruct-2512"
        tokenizer = MistralCommonBackend.from_pretrained(model_id)
        image_url = (
            "https://static.wikia.nocookie.net/essentialsdocs/images/7/70/Battle.png/revision/latest?cb=20220523172438"
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What action do you think I should take in this situation? List all the possible actions and explain why you think they are good or bad.",
                    },
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            },
        ]
        tokenized = tokenizer.apply_chat_template(messages, return_tensors="pt", return_dict=True)

        model = ExecuTorchModelForMultiModalToText.from_pretrained(
            model_id,
            recipe="xnnpack",
            task="multimodal-text-to-text",
            use_custom_sdpa=True,
            use_custom_kv_cache=True,
            qlinear="8da4w",
            qlinear_group_size=32,
            qlinear_encoder="8da4w",
            qlinear_encoder_group_size=32,
            qembedding="8w",
            qembedding_encoder="8w",
        )

        # Generate
        generated_text = model.text_generation(
            input_conversation=conversation,
            tokenizer=tokenizer,
            max_seq_len=64,
        )
        logging.info(f"\nGenerated text:\n\t{generated_text}")
        generated_tokens = tokenizer(generated_text, return_tensors="pt").input_ids
        breakpoint()

        # Free memory before loading eager for quality check
        del model
        del tokenizer
        gc.collect()

        self.assertTrue(check_causal_lm_output_quality(model_id, generated_tokens))
