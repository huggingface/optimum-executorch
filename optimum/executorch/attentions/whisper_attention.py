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

# Export friendly cross attention implementation for Whisper. Adopted
# from https://github.com/huggingface/transformers/blob/454c0a7ccf33f7fc13e3e2eb9b188a5c09ab708b/src/transformers/models/whisper/modeling_whisper.py#L241
# Rewritten to replace if branches with torch.cond. Note that unlike
# the original WhisperAttention, this implementation only works for
# cross attention (where `key_value_states` is not None).

from typing import Callable, Optional

import torch
from executorch.extension.llm.custom_ops import custom_ops  # noqa
from torch import Tensor, nn
from transformers.cache_utils import EncoderDecoderCache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.whisper.configuration_whisper import WhisperConfig
from transformers.models.whisper.modeling_whisper import eager_attention_forward
from transformers.processing_utils import Unpack
from transformers.utils import logging


logger = logging.get_logger(__name__)


class WhisperCrossAttention(nn.Module):
    """Multi-headed cross attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        is_causal: bool = False,
        layer_idx: Optional[int] = None,
        config: Optional[WhisperConfig] = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.config = config

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder
        self.is_causal = is_causal

        if layer_idx is None and is_decoder:
            logger.warning_once(
                f"Instantiating a decoder {self.__class__.__name__} without passing `layer_idx` is not recommended and "
                "will to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )
        self.layer_idx = layer_idx

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.register_buffer("cache_initialized", torch.zeros(1, 1, dtype=torch.bool), persistent=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: torch.Tensor,
        past_key_values: EncoderDecoderCache,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        cache_position: Optional[torch.Tensor] = None,
        # TODO: we need a refactor so that the different attention modules can get their specific kwargs
        # ATM, we have mixed things encoder, decoder, and encoder-decoder attn
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""
        torch._assert(
            isinstance(past_key_values, EncoderDecoderCache),
            f"past_key_values must be an EncoderDecoderCache, got {type(past_key_values)}",
        )
        # determine input shapes
        bsz, tgt_len = hidden_states.shape[:-1]
        q_input_shape = (bsz, tgt_len, -1, self.head_dim)

        # Scaling is susceptible to floating point arithmetics' inprecisions
        # which can lead to different results (this is dependent from model
        # to model, e.g. whisper is one such case). We therefore keep the
        # original order of scaling to follow the original implementation
        # and enforce no scaling (1.0) in the attention call below.
        query_states = self.q_proj(hidden_states) * self.scaling
        query_states = query_states.view(*q_input_shape)
        query_states = query_states.transpose(1, 2).contiguous()

        # Check is encoder-decoder model is being used. Otherwise we'll get `DynamicCache`
        if past_key_values is not None and isinstance(past_key_values, EncoderDecoderCache):
            # after the first generated id, we can subsequently re-use all key/value_states from cache
            past_key_values = past_key_values.cross_attention_cache

        def use_cached_kv(
            cached_keys: Tensor,
            cached_values: Tensor,
            key_value_states: Tensor,
        ) -> tuple[Tensor, Tensor]:
            # Just reuse cached K/V via custom op to avoid torch.cond aliasing trace errors
            return torch.ops.executorch.alias(cached_keys, cached_values)

        def recompute_kv(
            cached_keys: Tensor,  # unused
            cached_values: Tensor,  # unused
            key_value_states: Tensor,
        ) -> tuple[Tensor, Tensor]:
            # Compute fresh K/V (export-friendly: no cache mutation in here)
            # Use decomposed linear (matmul + add) instead of nn.Linear to avoid
            # aten.linear ops which XNNPACK partitioner can't delegate inside torch.cond
            # k_proj: no bias
            key_states = torch.matmul(key_value_states, self.k_proj.weight.t())
            key_states = key_states.view(bsz, -1, self.num_heads, self.head_dim)
            # v_proj: has bias
            value_states = torch.matmul(key_value_states, self.v_proj.weight.t())
            if self.v_proj.bias is not None:
                value_states = value_states + self.v_proj.bias
            value_states = value_states.view(bsz, -1, self.num_heads, self.head_dim)
            key_states = key_states.transpose(1, 2).contiguous()
            value_states = value_states.transpose(1, 2).contiguous()
            k = torch.ops.executorch.update_cross_attn_cache(key_states, cached_keys)
            v = torch.ops.executorch.update_cross_attn_cache(value_states, cached_values)
            return k, v

        # Grab cached tensors (these are Tensors, so they are OK for export)
        cached_keys = past_key_values.layers[self.layer_idx].keys
        cached_values = past_key_values.layers[self.layer_idx].values

        # Use torch.cond to select branch in a traceable way.
        # All operands must be (nested) tensors or simple Python values.
        key_states, value_states = torch.cond(
            self.cache_initialized,
            use_cached_kv,
            recompute_kv,
            operands=(cached_keys, cached_values, key_value_states),
        )

        # Update the cache_initialized flag to True after first use
        self.cache_initialized.fill_(True)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.dropout,
            scaling=1.0,
            output_attentions=output_attentions,
            **kwargs,
        )

        attn_output = attn_output.reshape(bsz, tgt_len, -1).contiguous()
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights
