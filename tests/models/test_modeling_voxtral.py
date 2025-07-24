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
from transformers import AutoTokenizer, AutoProcessor
from transformers.testing_utils import slow

from optimum.utils.import_utils import is_transformers_version
from optimum.exporters.executorch.tasks.multimodal_text_to_text import load_multimodal_text_to_text_model

from ..utils import check_causal_lm_output_quality


is_linux_ci = sys.platform.startswith("linux") and os.environ.get("GITHUB_ACTIONS") == "true"


os.environ["TOKENIZERS_PARALLELISM"] = "false"


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

    # @slow
    # @pytest.mark.run_slow
    # @pytest.mark.skipif(
    #     parse(transformers.__version__) < parse("4.53.0.dev0") or parse(torchao.__version__) < parse("0.11.0"),
    #     reason="Only available on transformers >= 4.53.0.dev0 and torchao >= 0.11.0",
    # )
    # @pytest.mark.skipif(is_linux_ci, reason="OOM on linux runner")
    def test_voxtral_audio_text_to_text_generation_with_custom_sdpa_kv_cache_8da4w_8we(self):
        model_id = "mistralai/Voxtral-Mini-3B-2507"
        module = load_multimodal_text_to_text_model(
            model_id,
            use_custom_sdpa=True,
            use_custom_kv_cache=True,
            qlinear=True,
            qembedding_config=True,
        )

        res = module.export()

        breakpoint()

        # Generate
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        image_url = "https://llava-vl.github.io/static/images/view.jpg"
        conversation = [
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": "You are a helpful assistant."}]
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "url": image_url},
                            {
                                "type": "text",
                                "text": "What are the things I should be cautious about when I visit here?",
                            },
                        ],
                    },
                ]
        processor = AutoProcessor.from_pretrained(model_id)
        inputs = processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True, 
            return_tensors="pt",
        )
        image_indices = torch.where(inputs["input_ids"] == module.model.model.config.image_token_id)
        prompt_before_image = inputs["input_ids"][:, :image_indices[1][0]]
        prompt_after_image = inputs["input_ids"][:, image_indices[1][-1]+1:]

        image_features = res["vision_embeddings"].module().forward(pixel_values=inputs["pixel_values"])

        print(prompt_before_image.shape)

        torch.arange(prompt_before_image.shape[1], device=inputs["input_ids"].device)

        token_embeddings_before_image = res["token_embeddings"].module().forward(
            input_ids=prompt_before_image)

        token_embeddings_after_image = res["token_embeddings"].module().forward(
            input_ids=prompt_after_image)

        embeddings = torch.cat(
            [
                token_embeddings_before_image,
                image_features,
                token_embeddings_after_image,
            ],
            dim=1,
        )

        print(embeddings.shape)

        # Prefill prompt embeddings
        logits = res["decoder"].module().forward(
            inputs_embeds=embeddings,
            cache_position=torch.arange(embeddings.shape[1], dtype=torch.long),
        )

        token = torch.argmax(logits[:, -1, :])

        tokens = [token.item()]

        pos = embeddings.shape[1]

        while pos < 350:
            token_embedding = res["token_embeddings"].module().forward(
                input_ids=token.unsqueeze(0).unsqueeze(0)
            )
            logits = res["decoder"].module().forward(
                inputs_embeds=token_embedding,
                cache_position=torch.tensor([pos], dtype=torch.long),
            )
            token = torch.argmax(logits[:, -1, :])
            tokens.append(token.item())
            pos += 1

        output = tokenizer.decode(tokens, skip_special_tokens=True)
        self.assertEqual(
            output,
            """Okay, let's analyze the image and discuss potential cautions for visiting this location. 

Based on the picture, we're looking at a serene lakeside scene with a wooden pier extending into the water. Here's a breakdown of what you should be cautious about, categorized for clarity:

**1""",
        )
