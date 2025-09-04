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
        config = AutoConfig.from_pretrained(model_id)
        module = load_multimodal_text_to_text_model(
            model_id,
            use_custom_sdpa=True,
            use_custom_kv_cache=True,
            qlinear="8da4w",
            qembedding="4w",
        )

        ep = module.export()

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
        token_embeddings = ep["token_embedding"].module().forward(input=input_ids)

        if "input_features" in inputs:
            audio_embeddings = (
                ep["audio_encoder"]
                .module()
                .forward(
                    input_features=inputs["input_features"],
                )
            )

        audio_token_mask = inputs["input_ids"] == config.audio_token_id
        token_embeddings[audio_token_mask] = audio_embeddings

        # Prefill prompt embeddings
        logits = (
            ep["text_decoder"]
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
            token_embedding = ep["token_embedding"].module().forward(input=token.unsqueeze(0).unsqueeze(0))
            logits = (
                ep["text_decoder"]
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
    def test_voxtral_audio_text_to_text_generation_with_custom_sdpa_kv_cache_8da4w_8we_split_prefill_exported_program(
        self,
    ):
        """
        Similar test as above by testing E2E with ExportedPrograms, but does so by splitting prefill
        into three parts instead of doing it all at once:
        - Special tokens denoting applied automatically to the prompt by the mistral tokenizer
          denoting start of prompt, start of modality, etc.
        - Audio embeddings
        - Prompt token embeddings

        This split-up prefill process mirrors how we do multimodal prefill in ET's multimodal runner.
        To compare, transformers does:
        - Format text prompt something like input_ids = ["<bos>", "audio_token",
        "audio_placeholder", ... , "audio_placeholder", "Hello", " ", "world", "<eos>"] in the
        VoxtralProcessor
        - Embed the input_ids
        - Run encoder and replace audio placeholders with audio embeddings with masking
        - Do one combined prefill with everything

        Splitting the prefill allows us to skip the custom logic of needing to format the
        input, including calculating how many audio placeholders to add, etc. Another advantage
        is that it allows us to do multiturn multimodal (submit multiple audios throughout an
        ongoing chat). Obviously doing a single prefill would be faster though.
        """
        model_id = "mistralai/Voxtral-Mini-3B-2507"
        _config = AutoConfig.from_pretrained(model_id)
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
        input_features = inputs["input_features"]

        # Load and torch.export model.
        module = load_multimodal_text_to_text_model(
            model_id,
            use_custom_sdpa=True,
            use_custom_kv_cache=True,
            qlinear="8da4w",
            qembedding="4w",
        )
        ep = module.export()

        # 1. Prefill start metadata tokens.
        cache_pos = 0
        start_metadata_tokens = input_ids[:, 0:3]  # Starts with [1, 3, 25].
        start_metadata_embeddings = ep["token_embedding"].module().forward(input=start_metadata_tokens)
        start_metadata_len = start_metadata_tokens.shape[1]
        logits = (
            ep["text_decoder"]
            .module()
            .forward(
                inputs_embeds=start_metadata_embeddings,
                cache_position=torch.arange(cache_pos, cache_pos + start_metadata_len, dtype=torch.long),
            )
        )
        cache_pos += start_metadata_len

        # 2. Prefill audio.
        if "input_features" in inputs:
            audio_embeddings = (
                ep["audio_encoder"]
                .module()
                .forward(
                    input_features=input_features,
                )
            )
        audio_embeddings_len = audio_embeddings.shape[1]
        logits = (
            ep["text_decoder"]
            .module()
            .forward(
                inputs_embeds=audio_embeddings,
                cache_position=torch.arange(cache_pos, cache_pos + audio_embeddings_len, dtype=torch.long),
            )
        )
        cache_pos += audio_embeddings_len

        # 3. Prefill text prompt embeddings
        prompt_start_index = start_metadata_len + audio_embeddings_len
        prompt_tokens = input_ids[:, prompt_start_index:]
        prompt_tokens_embeddings = ep["token_embedding"].module().forward(input=prompt_tokens)
        prompt_tokens_len = prompt_tokens.shape[1]
        logits = (
            ep["text_decoder"]
            .module()
            .forward(
                inputs_embeds=prompt_tokens_embeddings,
                cache_position=torch.arange(cache_pos, cache_pos + prompt_tokens_len, dtype=torch.long),
            )
        )
        cache_pos += prompt_tokens_len

        token = torch.argmax(logits[:, -1, :])

        tokens = [token.item()]
        print(tokenizer.decode([token.item()]), end="")

        pos = cache_pos  # Should be equivalent to input_ids.shape[1]
        assert cache_pos == input_ids.shape[1]

        max_generation_len = 64
        while pos < input_ids.shape[-1] + max_generation_len:
            token_embedding = ep["token_embedding"].module().forward(input=token.unsqueeze(0).unsqueeze(0))
            logits = (
                ep["text_decoder"]
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
            recipe="xnnpack",
            attn_implementation="custom_sdpa",
            use_custom_kv_cache=True,
            **{"qlinear": "8da4w", "qembedding": "4w", "task": "multimodal-text-to-text"},
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
