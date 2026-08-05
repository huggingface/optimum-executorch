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

import logging
from typing import Callable, Optional, Tuple, Union

import torch
from executorch.extension.llm.custom_ops.custom_ops import custom_sdpa  # noqa


_HALF_DTYPES = (torch.float16, torch.bfloat16)


def _custom_sdpa_traces_half() -> bool:
    """
    Whether `llama.custom_sdpa` can be traced with an f16/bf16 query.

    ExecuTorch only taught the op about half dtypes in 1.4; earlier versions assert float32 in the
    meta kernel, which surfaces as an opaque dynamo failure rather than something we can catch
    around the call site. Probe the meta kernel once instead of comparing versions, since source
    builds report versions such as `1.4.0a0+<sha>` that no version predicate can classify.
    """
    from torch._subclasses.fake_tensor import FakeTensorMode

    # A failing probe is an expected outcome, so don't let the meta kernel log its traceback.
    fake_tensor_logger = logging.getLogger("torch._subclasses.fake_tensor")
    previously_disabled = fake_tensor_logger.disabled
    fake_tensor_logger.disabled = True
    try:
        with FakeTensorMode():
            qkv = torch.empty(1, 1, 1, 8, dtype=torch.bfloat16)
            torch.ops.llama.custom_sdpa(
                qkv,
                qkv,
                qkv,
                start_pos=0,
                attn_mask=None,
                drpout_p=0.0,
                is_causal=False,
                scale=None,
            )
    except Exception:
        return False
    finally:
        fake_tensor_logger.disabled = previously_disabled
    return True


# Evaluated at import time: the probe traces the op, so it must not run inside an export trace.
_CUSTOM_SDPA_TRACES_HALF = _custom_sdpa_traces_half()


def _is_tracing(tensor: torch.Tensor) -> bool:
    """Whether we are building an exported graph rather than actually computing."""
    # `is_compiling` is constant-folded by dynamo (strict export); non-strict export never enters
    # dynamo but does hand the forward fake tensors.
    return torch.compiler.is_compiling() or isinstance(tensor, torch._subclasses.FakeTensor)


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

    input_dtype = query.dtype

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
            cache_position = kwargs.get("cache_position")
            if position_ids is not None:
                start_pos = position_ids[0][0].item()
            elif cache_position is not None:
                start_pos = cache_position[0].item()
            else:
                start_pos = 0
        else:
            start_pos = 0

    # Keep half dtypes only where they are known to work: inside an exported graph, running against
    # an ExecuTorch that supports them. Eager calls go through the AOT op library, whose f16/bf16
    # kernel needs a temp allocator it is never given, so it silently returns garbage. Everywhere
    # else, upcast to fp32 and cast back.
    if input_dtype in _HALF_DTYPES and not (_CUSTOM_SDPA_TRACES_HALF and _is_tracing(query)):
        query = query.to(torch.float32)
        key = key.to(torch.float32)
        value = value.to(torch.float32)

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
