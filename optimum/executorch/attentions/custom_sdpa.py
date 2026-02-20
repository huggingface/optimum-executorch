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

from typing import Callable, Optional, Tuple, Union

import torch
from executorch.extension.llm.custom_ops.custom_ops import custom_sdpa  # noqa


def sdpa_mask_passthrough(
    batch_size: int,
    cache_position: torch.Tensor,
    kv_length: int,
    kv_offset: int = 0,
    mask_function: Optional[Callable] = None,
    attention_mask: Optional[torch.Tensor] = None,
    local_size: Optional[int] = None,
    allow_is_causal_skip: bool = True,
    allow_torch_fix: bool = True,
    **kwargs,
) -> Optional[torch.Tensor]:
    """
    Pass-through for attention mask creation since it is never used:
    - For regular attention, the custom sdpa op in causal mode creates its own attention mask
    - For sliding window attention, the attention mask from the attention mask API is ditched and re-created during the attention API since it needs to know about cache internals

    Additionally, there were some vmap export issues with sliding window attention mask creation in Transformers.

    Args:
        batch_size (`int`):
            The batch size of the input sequence.
        cache_position (`torch.Tensor`):
            A tensor of shape (query_length,) indicating the current indices of the input sequence elements.
        kv_length (`int`):
            The size that the key and value states will have during the attention computation.
        kv_offset (`int`, optional):
            An optional offset to indicate at which first position the key and values states will refer to.
        mask_function (`Callable`):
            The mask factory function describing the mask pattern.
        attention_mask (`torch.Tensor`, optional):
            The 2D attention mask corresponding to padded tokens of shape (batch_size, number_of_seen_tokens+q_length)
        local_size (`int`, optional):
            The size of the local attention, if we do not use full attention. This is used only if `allow_is_causal_skip=True`
            to try to skip mask creation if possible.
        allow_is_causal_skip (`bool`, optional):
            Whether to allow to return `None` for the mask under conditions where we can use the `is_causal` argument in
            `torch.sdpa` instead. Default to `True`.
        allow_torch_fix (`bool`, optional):
            Whether to update the mask in case a query is not attending to any tokens, to solve a bug in torch's older
            versions. We need an arg to skip it when using eager. By default `True`.

    """
    return None


def custom_sdpa_with_start_pos_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Union[torch.Tensor, "BlockMask"],  # noqa
    position_ids: Optional[torch.Tensor] = None,
    scaling: Optional[float] = None,
    softcap: Optional[float] = None,
    head_mask: Optional[torch.Tensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, None]:
    # FA2 uses non-transposed inputs
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)

    # Convert the hell out of the inputs to fp32 and back
    input_dtype = query.dtype
    query = query.to(torch.float32)
    key = key.to(torch.float32)
    value = value.to(torch.float32)

    # Ignore the causal flag from kwargs but use the one in module
    kwargs.pop("is_causal", None)

    is_causal = module.is_causal
    if kwargs.get("is_sliding", False):
        is_causal = False
        attn_mask = attention_mask
        # start_pos is not important when using mask
        # instead of doing causal attention
        start_pos = 0
    else:
        attn_mask = None
        if is_causal:
            # Calculate the input pos from attention mask.
            # Branch out for float vs bool mask
            # assert attention_mask.dim() == 2, f"attention_mask must be a 2D matrix."
            assert (
                position_ids is not None
            ), "position_ids must be provided to find start position for causal attention"
            start_pos = position_ids[0][0].item()
        else:
            start_pos = 0

    output = torch.ops.llama.custom_sdpa(
        query,
        key,
        value,
        start_pos=start_pos,
        attn_mask=attn_mask,
        drpout_p=0.0,
        is_causal=is_causal,
        scale=scaling,
    )
    return output.to(input_dtype), None


def get_custom_sdpa_for_ring_kv_cache(
    exportable_module: torch.nn.Module,
) -> Callable:
    # lazy importing to avoid version dependent class definition
    from executorch import version

    try:
        from executorch.examples.models.llama.source_transformation.custom_kv_cache import (
            CustomRingKVCache,
        )
    except ImportError:
        raise ImportError(f"CustomRingKVCache not available in version {version.__version__} of ExecuTorch.")

    def _custom_sdpa_for_ring_kv_cache(
        module: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Union[torch.Tensor, "BlockMask"],  # noqa
        position_ids: Optional[torch.Tensor] = None,
        scaling: Optional[float] = None,
        softcap: Optional[float] = None,
        head_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, None]:
        is_sliding = getattr(module, "is_sliding", False)
        if is_sliding:
            # lazy import to avoid being in the optimum import path
            # for et <= 0.6.0 version
            from optimum.executorch.attentions.custom_kv_cache import ETCustomHybridCache

            layer_idx = module.layer_idx
            assert layer_idx is not None, "layer_idx is not set for sliding window attention."
            hybrid_cache = exportable_module.model.cache
            assert isinstance(hybrid_cache, ETCustomHybridCache), f"Expected HybridCache, got {type(hybrid_cache)}"
            ring_cache = hybrid_cache.get_layer_cache(layer_idx)
            assert isinstance(ring_cache, CustomRingKVCache), f"Expected CustomRingKVCache, got {type(ring_cache)}"
            input_pos = hybrid_cache.cache_position[0].item()
            seqlen = query.shape[2]
            attention_mask = ring_cache.create_causal_mask_for_ring_buffer(input_pos, seqlen)
            kwargs.update({"is_sliding": True})
            return custom_sdpa_with_start_pos_forward(
                module,
                query,
                key,
                value,
                attention_mask,
                position_ids,
                scaling,
                softcap,
                head_mask,
                **kwargs,
            )
        else:
            return custom_sdpa_with_start_pos_forward(
                module,
                query,
                key,
                value,
                attention_mask,
                position_ids,
                scaling,
                softcap,
                head_mask,
                **kwargs,
            )

    return _custom_sdpa_for_ring_kv_cache


def get_custom_sdpa_for_attention_sink(
    exportable_module: torch.nn.Module,
) -> Callable:
    """Create SDPA function for attention sink models.
    ALL layers use ring buffer mask with sink token preservation."""

    from optimum.executorch.attentions.custom_kv_cache import (
        CustomRingKVCacheWithSink,
        ETCustomAttentionSinkCache,
    )

    def _custom_sdpa_for_attention_sink(
        module: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Union[torch.Tensor, "BlockMask"],  # noqa
        position_ids: Optional[torch.Tensor] = None,
        scaling: Optional[float] = None,
        softcap: Optional[float] = None,
        head_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, None]:
        layer_idx = module.layer_idx
        assert layer_idx is not None, "layer_idx is not set."
        sink_cache = exportable_module.model.static_cache
        assert isinstance(sink_cache, ETCustomAttentionSinkCache), (
            f"Expected ETCustomAttentionSinkCache, got {type(sink_cache)}"
        )
        ring_cache = sink_cache.get_layer_cache(layer_idx)
        assert isinstance(ring_cache, CustomRingKVCacheWithSink), (
            f"Expected CustomRingKVCacheWithSink, got {type(ring_cache)}"
        )
        input_pos = sink_cache.cache_position[0].item()
        seqlen = query.shape[2]
        attention_mask = ring_cache.create_causal_mask_for_ring_buffer(input_pos, seqlen)
        kwargs.update({"is_sliding": True})
        return custom_sdpa_with_start_pos_forward(
            module,
            query,
            key,
            value,
            attention_mask,
            position_ids,
            scaling,
            softcap,
            head_mask,
            **kwargs,
        )

    return _custom_sdpa_for_attention_sink
