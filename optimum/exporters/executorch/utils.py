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

from typing import List, Optional, Set

import torch
from transformers import GenerationConfig, PretrainedConfig
from transformers.tokenization_utils import PreTrainedTokenizer


def save_config_to_constant_methods(
    config: PretrainedConfig,
    generation_config: Optional[GenerationConfig] = None,
    processor_config: Optional[dict] = None,
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
        "use_kv_cache": getattr(generation_config, "use_cache", None),
        "sliding_window": getattr(config, "sliding_window", None),
        "decoder_start_token_id": getattr(config, "decoder_start_token_id", None),
        "use_sdpa_with_kv_cache": "custom_sdpa" in config._attn_implementation,
    }

    # Safely access fields from generation_config if it exists
    if generation_config is not None:
        # Check for cache_config and its attributes
        cache_config = getattr(generation_config, "cache_config", None)
        if cache_config is not None:
            max_batch_size = cache_config.get("batch_size")
            max_seq_len = cache_config.get("max_cache_len")

            if max_batch_size is not None:
                metadata["get_max_batch_size"] = max_batch_size
            if max_seq_len is not None:
                metadata["get_max_seq_len"] = max_seq_len

    # Include processor_config keys in metadata if provided
    if processor_config is not None:
        metadata.update(processor_config)

    # Combine with any additional kwargs and filter out None values
    return {k: v for k, v in {**metadata, **kwargs}.items() if v is not None}


def verify_eos_tokens_in_pretrained_tokenizer(model_eos_ids: List[int], tokenizer: PreTrainedTokenizer) -> bool:
    """
    Verifies that the model's EOS token IDs are present in the tokenizer's
    set of potential end-of-sequence tokens.

    Args:
        model_eos_ids: A list of EOS token IDs recorded int the PTE file (the source of truth).
        tokenizer: The Hugging Face tokenizer instance to check.

    Returns:
        True if at least one model EOS ID is found among the tokenizer's potential
        EOS tokens, False otherwise.
    """
    if not model_eos_ids:
        print("Warning: model_eos_ids list is empty. No verification can be performed.")
        return True

    candidate_eos_ids: Set[int] = set()

    # 1. Check primary eos_token and pad_token attributes
    if tokenizer.eos_token_id is not None:
        candidate_eos_ids.add(tokenizer.eos_token_id)
    if tokenizer.pad_token_id is not None:
        candidate_eos_ids.add(tokenizer.pad_token_id)

    # 2. Check all tokens listed in the special_tokens_map
    for token_string in tokenizer.special_tokens_map.values():
        if token_string:
            # Use convert_tokens_to_ids for robustness
            token_id = tokenizer.convert_tokens_to_ids(token_string)
            if isinstance(token_id, int):
                candidate_eos_ids.add(token_id)

    # 3. Check added tokens for "end-of-X" patterns
    for token_id, added_token in tokenizer.added_tokens_decoder.items():
        token_str = added_token.content.lower()
        # Heuristic to find tokens that signify an end
        if "end" in token_str or token_str.startswith("</"):
            candidate_eos_ids.add(token_id)

    # The check: is any "true" ID present in the candidate set?
    is_valid = any(model_id in candidate_eos_ids for model_id in model_eos_ids)

    return is_valid
