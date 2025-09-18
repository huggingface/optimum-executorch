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
from executorch.extension.pybindings.portable_lib import ExecuTorchModule
from transformers import AutoProcessor, AutoTokenizer
from transformers.testing_utils import slow

from optimum.executorch import ExecuTorchModelForMultiModalToText

from ..utils import check_multimodal_output_quality


is_linux_ci = sys.platform.startswith("linux") and os.environ.get("GITHUB_ACTIONS") == "true"

logging.basicConfig(level=logging.DEBUG)


class ExecuTorchModelIntegrationTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @slow
    @pytest.mark.run_slow
    @pytest.mark.skipif(is_linux_ci, reason="OOM")
    def test_granite_audio_text_to_text_generation_with_custom_sdpa_kv_cache_8da4w_8we_pte(self):
        model_id = "ibm-granite/granite-speech-3.3-2b"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        processor = AutoProcessor.from_pretrained(model_id)
        system_prompt = "Knowledge Cutoff Date: April 2024.\nToday's Date: April 9, 2025.\nYou are Granite, developed by IBM. You are a helpful AI assistant"
        user_prompt = "<|audio|>can you transcribe the speech into a written format?"
        conversation = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {
                "role": "user",
                "type": "audio",
                "content": "https://huggingface.co/ibm-granite/granite-speech-3.3-2b/resolve/main/10226_10111_000000.wav",
            },
        ]

        model = ExecuTorchModelForMultiModalToText.from_pretrained(
            # model_id,
            "/home/jackzhxng/models/granite/granite_1",
            recipe="xnnpack",
            attn_implementation="custom_sdpa",
            use_custom_kv_cache=True,
            **{
                "qlinear": "8da4w",
                "qlinear_encoder": "8da4w",
                "qembedding": "4w",
                "qembedding_group_size": 32,
                "task": "multimodal-text-to-text",
            },
        )
        self.assertIsInstance(model, ExecuTorchModelForMultiModalToText)
        self.assertIsInstance(model.model, ExecuTorchModule)

        generated_text = model.text_generation(
            processor=processor,
            tokenizer=tokenizer,
            input_conversation=conversation,
            max_seq_len=64,
        )
        logging.info(f"\nGenerated text:\n\t{generated_text}")
        generated_tokens = tokenizer(generated_text, return_tensors="pt").input_ids

        del model
        del tokenizer
        gc.collect()

        # Should be something like: 'Certainly! Here's the transcribed written format of Timothy's actions and thoughts:
        # After his nap, Timothy leisurely stretched, first one gray velvet foot, then the other. He then slowly rolled,
        # indolently, to his plate.'
        self.assertTrue("Timothy" in generated_text)
        self.assertTrue("nap" in generated_text)
        self.assertTrue("stretch" in generated_text)
        self.assertTrue(
            check_multimodal_output_quality(model_id, generated_tokens, conversation, max_perplexity_threshold=5)
        )
