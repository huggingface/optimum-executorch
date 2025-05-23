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

from typing import Optional

import torch
from transformers import GenerationConfig, PretrainedConfig


def save_config_to_constant_methods(
    config: PretrainedConfig,
    generation_config: Optional[GenerationConfig] = None,
    **kwargs,
):
    # Initialize metadata with values from model config
    head_dim = None
    if (
        hasattr(config, "hidden_size")
        and hasattr(config, "num_attention_heads")
        and isinstance(config.num_attention_heads, int)
    ):
        head_dim = config.hidden_size / config.num_attention_heads

    metadata = {
        "get_dtype": 5 if config.torch_dtype == torch.float16 else 6,
        "get_bos_id": getattr(config, "bos_token_id", None),
        "get_eos_id": getattr(config, "eos_token_id", None),
        "get_head_dim": head_dim,
        "get_n_kv_heads": getattr(config, "num_key_value_heads", None),
        "get_n_layers": getattr(config, "num_hidden_layers", None),
        "get_vocab_size": getattr(config, "vocab_size", None),
        "get_max_batch_size": 1,
        "get_max_seq_len": getattr(config, "max_position_embeddings", None),
        "decoder_start_token_id": getattr(config, "decoder_start_token_id", None),
        "use_sdpa_with_kv_cache": "custom_sdpa" in config._attn_implementation,
    }

    # Safely access fields from generation_config if it exists
    if generation_config is not None:
        # Get use_cache with default value
        use_cache = getattr(generation_config, "use_cache", None)
        metadata["use_kv_cache"] = use_cache

        # Check for cache_config and its attributes
        cache_config = getattr(generation_config, "cache_config", None)
        if cache_config is not None:
            max_batch_size = getattr(cache_config, "batch_size", None)
            max_seq_len = getattr(cache_config, "max_cache_len", None)

            if max_batch_size is not None:
                metadata["get_max_batch_size"] = max_batch_size
            if max_seq_len is not None:
                metadata["get_max_seq_len"] = max_seq_len

    # Combine with any additional kwargs and filter out None values
    return {k: v for k, v in {**metadata, **kwargs}.items() if v is not None}
