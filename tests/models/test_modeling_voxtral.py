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
import subprocess
import sys
import tempfile
import unittest

import pytest
import torch
import torchao
import transformers
from executorch.extension.pybindings.portable_lib import ExecuTorchModule
from packaging.version import parse
from transformers import AutoConfig, AutoTokenizer, AutoProcessor
from transformers.testing_utils import slow

from optimum.utils.import_utils import is_transformers_version
from optimum.executorch import ExecuTorchModelForMultiModalToText
from optimum.exporters.executorch.tasks.multimodal_text_to_text import load_multimodal_text_to_text_model

from ..utils import check_causal_lm_output_quality


is_linux_ci = sys.platform.startswith("linux") and os.environ.get("GITHUB_ACTIONS") == "true"


os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.basicConfig(level=logging.DEBUG)


@pytest.mark.skipif(
    is_transformers_version("<", "4.52.0.dev0"),
    reason="Only available on transformers >= 4.52.0.dev0",
)
class ExecuTorchModelIntegrationTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Register custom SDPA, which is usually registered in the convert script.
        from transformers.modeling_utils import AttentionInterface
        from optimum.executorch.attentions.custom_sdpa import custom_sdpa_with_start_pos_forward
        
        AttentionInterface.register("custom_sdpa", custom_sdpa_with_start_pos_forward)
        if is_transformers_version(">=", "4.53.0.dev0"):
            from transformers.integrations.executorch import sdpa_mask_without_vmap
            from transformers.masking_utils import AttentionMaskInterface
    
            AttentionMaskInterface.register("custom_sdpa", sdpa_mask_without_vmap)

    @slow
    @pytest.mark.run_slow
    @pytest.mark.skipif(
        parse(transformers.__version__) < parse("4.53.0.dev0") or parse(torchao.__version__) < parse("0.11.0"),
        reason="Only available on transformers >= 4.53.0.dev0 and torchao >= 0.11.0",
    )
    @pytest.mark.skipif(is_linux_ci, reason="OOM on linux runner")
    def test_voxtral_audio_text_to_text_generation_with_custom_sdpa_kv_cache_8da4w_8we_exported_program(self):
        model_id = "mistralai/Voxtral-Mini-3B-2507"
        config = AutoConfig.from_pretrained(model_id)
        module = load_multimodal_text_to_text_model(
            model_id,
            use_custom_sdpa=True,
            use_custom_kv_cache=True,
            qlinear=True,
            qembedding=True,
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
        token_embeddings = res["token_embeddings"].module().forward(
            input=input_ids)

        if "input_features" in inputs:
            token_embeddings = res["audio_encoder"].module().forward(
                input_features=inputs["input_features"],
                inputs_embeds=token_embeddings,
                input_ids=inputs["input_ids"],
            )

        # Prefill prompt embeddings
        logits = res["decoder"].module().forward(
            inputs_embeds=token_embeddings,
            cache_position=torch.arange(token_embeddings.shape[1], dtype=torch.long),
        )

        token = torch.argmax(logits[:, -1, :])

        tokens = [token.item()]
        print(tokenizer.decode([token.item()]), end="")

        pos = token_embeddings.shape[1]

        while pos < 2000:
            token_embedding = res["token_embeddings"].module().forward(
                input=token.unsqueeze(0).unsqueeze(0)
            )
            logits = res["decoder"].module().forward(
                inputs_embeds=token_embedding,
                cache_position=torch.tensor([pos], dtype=torch.long),
            )
            token = torch.argmax(logits[:, -1, :])
            print(tokenizer.decode([token.item()]), end="")
            tokens.append(token.item())
            pos += 1
            # TODO(JZ): end early.

        output = tokenizer.decode(tokens, skip_special_tokens=True)
        self.assertTrue(output.startswith("The audio features a conversation between two individuals, likely friends or acquaintances, who are discussing a series of tattoos."))

    @slow
    @pytest.mark.run_slow
    @pytest.mark.skipif(
        parse(transformers.__version__) < parse("4.53.0.dev0") or parse(torchao.__version__) < parse("0.11.0"),
        reason="Only available on transformers >= 4.53.0.dev0 and torchao >= 0.11.0",
    )
    @pytest.mark.skipif(is_linux_ci, reason="OOM on linux runner")
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
            **{"qlinear": True, "qembedding": True, "task": "multimodal-text-to-text"},
        )
        self.assertIsInstance(model, ExecuTorchModelForMultiModalToText)
        self.assertIsInstance(model.model, ExecuTorchModule)

        generated_text = model.text_generation(
            processor=processor,
            tokenizer=tokenizer,
            input_conversation=conversation,
            max_seq_len=64,
        )
        print(generated_text)
        breakpoint()

