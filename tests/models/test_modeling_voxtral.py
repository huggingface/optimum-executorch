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
import torch
from executorch.extension.pybindings.portable_lib import ExecuTorchModule
from transformers import AutoConfig, AutoProcessor, AutoTokenizer
from transformers.testing_utils import slow

from optimum.executorch import ExecuTorchModelForMultiModalToText
from optimum.exporters.executorch.tasks.multimodal_text_to_text import load_multimodal_text_to_text_model

from ..utils import check_multimodal_output_quality


is_linux_ci = sys.platform.startswith("linux") and os.environ.get("GITHUB_ACTIONS") == "true"


os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.basicConfig(level=logging.DEBUG)


class ExecuTorchModelIntegrationTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @slow
    @pytest.mark.run_slow
    @pytest.mark.skipif(is_linux_ci, reason="OOM")
    def test_voxtral_audio_text_to_text_generation_with_custom_sdpa_kv_cache_8da4w_8we_exported_program(self):
        """
        This test seems kind of unnecessary since we have test_voxtral_audio_text_to_text_generation_with_custom_sdpa_kv_cache_8da4w_8we_pte which just directly tests the excecutorch program, but keeping this here in case it's useful for showcasing code / debugging later on.
        """
        model_id = "mistralai/Voxtral-Mini-3B-2507"
        _config = AutoConfig.from_pretrained(model_id)
        module = load_multimodal_text_to_text_model(
            model_id,
            use_custom_sdpa=True,
            use_custom_kv_cache=True,
            qlinear="8da4w",
            qembedding="8w",
        )

        res = module.export()

        # Generate
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        conversation = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "audio",
                        "url": "https://huggingface.co/datasets/eustlb/audio-samples/resolve/main/dude_where_is_my_car.wav",
                    },
                    {"type": "text", "text": "What can you tell me about this audio?"},
                ],
            }
        ]
        processor = AutoProcessor.from_pretrained(model_id)
        inputs = processor.apply_chat_template(conversation)

        input_ids = inputs["input_ids"]
        token_embeddings = res["token_embeddings"].module().forward(input=input_ids)

        if "input_features" in inputs:
            token_embeddings = (
                res["audio_encoder"]
                .module()
                .forward(
                    input_features=inputs["input_features"],
                    inputs_embeds=token_embeddings,
                    input_ids=inputs["input_ids"],
                )
            )

        # Prefill prompt embeddings
        logits = (
            res["decoder"]
            .module()
            .forward(
                inputs_embeds=token_embeddings,
                cache_position=torch.arange(token_embeddings.shape[1], dtype=torch.long),
            )
        )

        token = torch.argmax(logits[:, -1, :])

        tokens = [token.item()]
        print(tokenizer.decode([token.item()]), end="")

        pos = token_embeddings.shape[1]

        max_generation_len = 64
        while pos < input_ids.shape[-1] + max_generation_len:
            token_embedding = res["token_embeddings"].module().forward(input=token.unsqueeze(0).unsqueeze(0))
            logits = (
                res["decoder"]
                .module()
                .forward(
                    inputs_embeds=token_embedding,
                    cache_position=torch.tensor([pos], dtype=torch.long),
                )
            )
            token = torch.argmax(logits[:, -1, :])
            print(tokenizer.decode([token.item()]), end="")
            tokens.append(token.item())
            pos += 1

        output = tokenizer.decode(tokens, skip_special_tokens=True)
        self.assertTrue("tattoo" in output)

    @slow
    @pytest.mark.run_slow
    @pytest.mark.skipif(is_linux_ci, reason="OOM")
    def test_voxtral_audio_text_to_text_generation_with_custom_sdpa_kv_cache_8da4w_8we_pte(self):
        model_id = "mistralai/Voxtral-Mini-3B-2507"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        processor = AutoProcessor.from_pretrained(model_id)
        conversation = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "audio",
                        "url": "https://huggingface.co/datasets/eustlb/audio-samples/resolve/main/dude_where_is_my_car.wav",
                    },
                    {"type": "text", "text": "What can you tell me about this audio?"},
                ],
            }
        ]

        model = ExecuTorchModelForMultiModalToText.from_pretrained(
            model_id,
            # "/home/jackzhxng/models/voxtral/voxtral_q_edm_qe_d",
            recipe="xnnpack",
            attn_implementation="custom_sdpa",
            use_custom_kv_cache=True,
            **{"qlinear": "8da4w", "qembedding": "8w", "task": "multimodal-text-to-text"},
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
        # Should be something like: 'The audio is a humorous conversation between two people,
        # likely friends or acquaintances, who are discussing tattoos.'

        del model
        del tokenizer
        gc.collect()

        self.assertTrue("tattoo" in generated_text)
        self.assertTrue(check_multimodal_output_quality(model_id, generated_tokens, conversation))
