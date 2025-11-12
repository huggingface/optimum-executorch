# Copyright (c) Meta Platforms, Inc. and affiliates.
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
import os
from typing import Dict, Union

import torch

from executorch.devtools.backend_debug import get_delegation_info
from executorch.exir import (
    EdgeCompileConfig,
    ExecutorchProgram,
    to_edge_transform_and_lower,
)

# from optimized_sdpa_triton import optimized_triton_scaled_dot_product_attention
from optimum.executorch.passes.remove_padding_idx_embedding_pass import (
    RemovePaddingIdxEmbeddingPass,
)
from tabulate import tabulate
from torch.export import ExportedProgram
from torch.nn.attention import SDPBackend

from ..integrations import (
    CausalLMExportableModule,
    MaskedLMExportableModule,
    MultiModalTextToTextExportableModule,
    Seq2SeqLMExportableModule,
)
from ..recipe_registry import register_recipe


aten = torch.ops.aten


import math
from typing import Any, Optional

import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton


def _validate_qkv_shapes(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> tuple[int, int, int, int, int, int]:
    """
    Validate dimensions and return shape info.
    Args:
        query: Query tensor [B, H, L_q, D]
        key: Key tensor [B, H, L_kv, D]
        value: Value tensor [B, H, L_kv, D]
    Returns:
        Tuple of (B, H, L_q, L_kv, D_q, D_kv)
    Raises:
        RuntimeError: If dimensions are incompatible
    """
    B_q, H_q, L_q, D_q = query.shape
    B_k, H_k, L_kv_k, D_k = key.shape
    B_v, H_v, L_kv_v, D_v = value.shape
    # Validate batch and head dimensions
    if not (B_q == B_k == B_v):
        raise RuntimeError(
            f"Batch dimension must match; got B_q={B_q}, B_k={B_k}, B_v={B_v}."
        )

    if not (H_q == H_k == H_v):
        raise RuntimeError(
            f"Head dimension must match; got H_q={H_q}, H_k={H_k}, H_v={H_v}."
        )
    # Head dimension must match
    if not (D_q == D_k == D_v):
        raise RuntimeError(
            f"Head dimension must match across Q, K, V; got D_q={D_q}, D_k={D_k}, D_v={D_v}."
        )
    # Key and Value sequence lengths must match
    if L_kv_k != L_kv_v:
        raise RuntimeError(
            f"Key and Value must have the same sequence length; got L_k={L_kv_k}, L_v={L_kv_v}."
        )
    return B_q, H_q, L_q, L_kv_k, D_q, D_k


@triton.autotune(
    configs=[
        # triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_stages=4, num_warps=8),
        # triton.Config({"BLOCK_M": 128, "BLOCK_N": 256}, num_stages=4, num_warps=8),
        # triton.Config({"BLOCK_M": 64, "BLOCK_N": 256}, num_stages=4, num_warps=4),
        # triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_stages=4, num_warps=4),
        # triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_stages=3, num_warps=4),
    ],
    key=["L_Q", "L_KV", "HEAD_DIM"],
)
@triton.jit
def _sdpa_fwd_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    mask_ptr,
    o_ptr,
    B,
    H,
    L_Q,  # Query sequence length
    L_KV,  # Key/Value sequence length
    HEAD_DIM,
    stride_qb,
    stride_qh,
    stride_ql,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_kl,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vl,
    stride_vd,
    stride_mb,
    stride_mh,
    stride_ml,
    stride_mn,
    stride_ob,
    stride_oh,
    stride_ol,
    stride_od,
    sm_scale,
    IS_CAUSAL: tl.constexpr,
    HAS_MASK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM_CE: tl.constexpr,
):
    """
    Fused SDPA kernel that handles different sequence lengths for Q and K/V.

    Q shape: [B, H, L_Q, D]
    K/V shape: [B, H, L_KV, D]
    Output shape: [B, H, L_Q, D]
    """
    # Program IDs
    pid_m = tl.program_id(axis=0)  # along query length
    pid_hz = tl.program_id(axis=1)  # flattened batch*head
    off_b = pid_hz // H
    off_h = pid_hz % H
    # Compute ranges for queries
    start_m = pid_m * BLOCK_M
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM_CE)
    mask_m = offs_m < L_Q  # Mask based on query length
    # Base pointers for this (b, h)
    q_base = q_ptr + off_b * stride_qb + off_h * stride_qh
    k_base = k_ptr + off_b * stride_kb + off_h * stride_kh
    v_base = v_ptr + off_b * stride_vb + off_h * stride_vh
    o_base = o_ptr + off_b * stride_ob + off_h * stride_oh
    # Mask base pointer (if provided)
    if HAS_MASK:
        mask_base = mask_ptr + off_b * stride_mb + off_h * stride_mh
    # Make head-dim addresses compiler-friendly
    offs_d_ctg = tl.max_contiguous(tl.multiple_of(offs_d, 16), HEAD_DIM_CE)
    # Load Q tile [BLOCK_M, HEAD_DIM] - coalesced along HEAD_DIM
    q_ptrs = q_base + (offs_m[:, None] * stride_ql + offs_d_ctg[None, :] * stride_qd)
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
    q = q.to(tl.bfloat16)
    # Initialize accumulators and softmax stats
    acc = tl.zeros((BLOCK_M, HEAD_DIM_CE), dtype=tl.float32)
    m_i = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    # Convert to base-2 scale for exp2
    qk_scale = sm_scale * 1.4426950408889634
    # Loop over keys/values along L_KV dimension (not L_Q!)
    for start_n in tl.range(0, L_KV, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < L_KV  # Mask based on key/value length
        # Load K tile [BLOCK_N, HEAD_DIM] (contiguous along HEAD_DIM)
        k_ptrs = k_base + (
            offs_n[:, None] * stride_kl + offs_d_ctg[None, :] * stride_kd
        )
        k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
        k = k.to(tl.bfloat16)
        # Compute attention logits [BLOCK_M, BLOCK_N] = Q[BM,D] @ K[BN,D]^T
        qk = tl.dot(q, tl.trans(k)).to(tl.float32)
        qk = qk * qk_scale
        # Apply causal mask if needed
        # For causal masking with different lengths: position i can attend to position j if i >= j
        if IS_CAUSAL:
            causal_mask = offs_m[:, None] >= offs_n[None, :]
            qk = tl.where(causal_mask, qk, -float("inf"))
        # Apply attention mask if provided
        if HAS_MASK:
            # Load mask tile [BLOCK_M, BLOCK_N]
            # Mask shape should be [B, H, L_Q, L_KV]
            mask_ptrs = mask_base + (
                offs_m[:, None] * stride_ml + offs_n[None, :] * stride_mn
            )
            attn_mask = tl.load(
                mask_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0
            )
            # Convert boolean mask to additive mask (-inf for False, 0 for True)
            qk = tl.where(attn_mask, qk, -float("inf"))
        # Apply OOB masks for both rows and cols
        qk = tl.where(mask_n[None, :], qk, -float("inf"))
        qk = tl.where(mask_m[:, None], qk, -float("inf"))
        # Online softmax
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.math.exp2(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp2(m_i - m_ij)
        # Load V tile [BLOCK_N, HEAD_DIM] (contiguous along HEAD_DIM)
        v_ptrs = v_base + (
            offs_n[:, None] * stride_vl + offs_d_ctg[None, :] * stride_vd
        )
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
        v = v.to(tl.bfloat16)
        # Update accumulator
        acc = acc * alpha[:, None]
        p_bf16 = p.to(tl.bfloat16)
        acc = tl.dot(p_bf16, v, acc)
        # Update softmax stats
        l_i = l_i * alpha + l_ij
        m_i = m_ij
    # Normalize accumulator by softmax denominator
    acc = acc / l_i[:, None]
    # Store output [BLOCK_M, HEAD_DIM] - shape matches query
    o_ptrs = o_base + (offs_m[:, None] * stride_ol + offs_d_ctg[None, :] * stride_od)
    tl.store(o_ptrs, acc.to(tl.bfloat16), mask=mask_m[:, None])


@triton_op("custom::optimized_triton_sdpa", mutates_args={})
def optimized_triton_scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float = 0.0,
    enable_gqa: bool = False,
) -> torch.Tensor:
    """
    Triton fused Scaled Dot-Product Attention with support for different sequence lengths.

    Supports different sequence lengths for query and key/value:
    - Query: [B, H, L_q, D]
    - Key: [B, H, L_kv, D]
    - Value: [B, H, L_kv, D]
    - Output: [B, H, L_q, D] (matches query shape)
    Args:
        query: Query tensor [B, H, L_q, D]
        key: Key tensor [B, H, L_kv, D]
        value: Value tensor [B, H, L_kv, D]
        attn_mask: Optional attention mask [B, H, L_q, L_kv] or broadcastable shape
        dropout_p: must be 0.0 (not supported)
        is_causal: whether to apply causal masking
        scale: attention scale (default: 1/sqrt(d))
        enable_gqa: must be False (not supported)
    Returns:
        Output tensor [B, H, L_q, D]
    """
    # Validate inputs
    if not (query.is_cuda and key.is_cuda and value.is_cuda):
        raise RuntimeError("Q, K, V must be CUDA tensors.")
    if (
        query.dtype != torch.bfloat16
        or key.dtype != torch.bfloat16
        or value.dtype != torch.bfloat16
    ):
        raise RuntimeError("Expected bfloat16 inputs")
    if query.dim() != 4 or key.dim() != 4 or value.dim() != 4:
        raise RuntimeError(
            f"Expected 4D tensors shaped [B, H, L, D]; got query.dim()={query.dim()}, key.dim()={key.dim()}, value.dim()={value.dim()}."
        )
    # Enforce unsupported features
    if dropout_p != 0.0:
        raise RuntimeError(
            "dropout_p must be 0.0 (not supported in this implementation)."
        )
    if enable_gqa is not False:
        raise RuntimeError(
            "enable_gqa must be False (not supported in this implementation)."
        )
    # Validate and get dimensions
    B, H, L_q, L_kv, D_q, D_kv = _validate_qkv_shapes(query, key, value)
    D = D_q  # Head dimension
    # Allocate output with query shape
    out = torch.empty_like(query)
    # Element-wise strides
    sqb, sqh, sql, sqd = query.stride()
    skb, skh, skl, skd = key.stride()
    svb, svh, svl, svd = value.stride()
    sob, soh, sol, sod = out.stride()

    # Grid: tile queries (M) and batch*heads axis
    def grid(META):
        return (
            triton.cdiv(L_q, META["BLOCK_M"]),  # Based on query length
            B * H,
        )

    # Scale factor for SDPA
    sm_scale = 1.0 / math.sqrt(D) if scale == 0.0 else scale
    # Handle attention mask
    has_mask = attn_mask is not None
    if has_mask:
        # Expand mask to [B, H, L_q, L_kv] if needed
        if attn_mask.dim() == 2:
            # [L_q, L_kv] -> [B, H, L_q, L_kv]
            attn_mask = attn_mask.unsqueeze(0).unsqueeze(0).expand(B, H, -1, -1)
        elif attn_mask.dim() == 3:
            # [B, L_q, L_kv] -> [B, H, L_q, L_kv]
            attn_mask = attn_mask.unsqueeze(1).expand(-1, H, -1, -1)

        # Validate mask shape
        if attn_mask.shape != (B, H, L_q, L_kv):
            # Try to expand if broadcastable
            attn_mask = attn_mask.expand(B, H, L_q, L_kv)

        smb, smh, sml, smn = attn_mask.stride()
    else:
        # Dummy strides and mask
        smb, smh, sml, smn = 0, 0, 0, 0
        attn_mask = torch.empty(0, dtype=torch.bool, device=query.device)
    # Launch kernel
    wrap_triton(_sdpa_fwd_kernel)[grid](
        query,
        key,
        value,
        attn_mask,
        out,
        B,
        H,
        L_q,  # Query sequence length
        L_kv,  # Key/Value sequence length
        D,
        sqb,
        sqh,
        sql,
        sqd,
        skb,
        skh,
        skl,
        skd,
        svb,
        svh,
        svl,
        svd,
        smb,
        smh,
        sml,
        smn,
        sob,
        soh,
        sol,
        sod,
        sm_scale,
        IS_CAUSAL=is_causal,
        HAS_MASK=has_mask,
        HEAD_DIM_CE=D,
    )
    return out


# Register the abstract/fake implementation for torch.export
# This is critical to avoid accessing real tensor data during export
@optimized_triton_scaled_dot_product_attention.register_fake
def _optimized_triton_sdpa_abstract(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale=None,
    enable_gqa=False,
) -> torch.Tensor:
    """
    Abstract/fake implementation for torch.export.
    This just returns an empty tensor with the correct shape/dtype/device.
    No actual computation happens here - this is only for shape inference during export.
    Supports different sequence lengths for query and key/value.
    Output shape matches query shape (not broadcasted).
    """
    # Validate dtypes match
    assert query.dtype == key.dtype == value.dtype, "Q, K, V must have the same dtype"
    # Get the output shape - should match query shape
    # This is important for fake mode - we only need to infer shapes, not create tensors
    B, H, L_q, _, D_q, _ = _validate_qkv_shapes(query, key, value)
    # Output shape matches query shape (following SDPA semantics)
    # IMPORTANT: Use the exact same dtype to satisfy ExecuTorch validation
    return torch.empty(B, H, L_q, D_q, dtype=query.dtype, device=query.device)


@register_recipe("cuda")
def export_to_executorch_with_cuda(
    model: Union[
        CausalLMExportableModule,
        MaskedLMExportableModule,
        Seq2SeqLMExportableModule,
        MultiModalTextToTextExportableModule,
    ],
    **kwargs,
):
    """
    Export a PyTorch model to ExecuTorch w/ delegation to CUDA backend.
    This function also write metadata required by the ExecuTorch runtime to the .pte file.
    Args:
        model (Union[CausalLMExportableModule, MaskedLMExportableModule, Seq2SeqLMExportableModule, MultiModalTextToTextExportableModule]):
            The PyTorch model to be exported to ExecuTorch.
        **kwargs:
            Additional keyword arguments for recipe-specific configurations, e.g. export using different example inputs, or different compile/bechend configs.
    Returns:
        Dict[str, ExecutorchProgram]:
            A map of exported and optimized program for ExecuTorch.
            For encoder-decoder models or multimodal models, it may generate multiple programs.
    """
    from executorch.backends.cuda.cuda_backend import CudaBackend
    from executorch.backends.cuda.cuda_partitioner import CudaPartitioner

    # Import here to avoid version conflicts.
    from torch._inductor.decomposition import conv1d_to_conv2d

    # with torch.nn.attention.sdpa_kernel([SDPBackend.MATH]), torch.no_grad():

    def _lower_to_executorch(
        exported_programs: Dict[str, ExportedProgram],
        metadata=None,
    ) -> Dict[str, ExecutorchProgram]:
        logging.debug(f"\nExported program: {exported_programs}")

        # If just one exported program, the method name in the .pte for it should be "forward".
        if len(exported_programs) == 1:
            exported_programs = {"forward": next(iter(exported_programs.values()))}

        # CUDA backend compile spec with method name.
        partitioners = {
            key: [CudaPartitioner([CudaBackend.generate_method_name_compile_spec(key)])]
            for key in exported_programs.keys()
        }
        # Add decompositions for triton to generate kernels.
        for key, ep in exported_programs.items():
            exported_programs[key] = ep.run_decompositions(
                {
                    aten.conv1d.default: conv1d_to_conv2d,
                }
            )
        # with torch.nn.attention.sdpa_kernel([SDPBackend.MATH]):
        et_prog = to_edge_transform_and_lower(
            exported_programs,
            partitioner=partitioners,
            compile_config=EdgeCompileConfig(
                _check_ir_validity=False,
                _skip_dim_order=True,
            ),
            constant_methods=metadata,
            transform_passes=[RemovePaddingIdxEmbeddingPass()],
        )
        et_prog = et_prog.to_executorch()
        pte_name = "model"
        for method in et_prog.methods:
            logging.debug(
                f"---------------------- Method: {method} ----------------------"
            )
            logging.debug(
                f"\nExecuTorch program for {pte_name}.pte: {et_prog.exported_program(method).graph_module}"
            )
            delegation_info = get_delegation_info(
                et_prog.exported_program(method).graph_module
            )
            logging.debug(
                f"\nDelegation info Summary for {pte_name}.pte: {delegation_info.get_summary()}"
            )
            logging.debug(
                f"\nDelegation info for {pte_name}.pte: {tabulate(delegation_info.get_operator_delegation_dataframe(), headers='keys', tablefmt='fancy_grid')}"
            )
        return {pte_name: et_prog}

    # Wrap the CustomOpDef in a regular function to avoid __name__ attribute error
    # during torch.export when PyTorch tries to check overridable functions
    def _wrapped_sdpa(
        query,
        key,
        value,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
        scale=None,
        enable_gqa=False,
    ):
        return optimized_triton_scaled_dot_product_attention(
            query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa
        )

    torch.nn.functional.scaled_dot_product_attention = _wrapped_sdpa

    if (
        model.config._attn_implementation == "custom_sdpa"
        or model.config._attn_implementation == "custom_sdpa_ring_kv_cache"
    ):
        raise NotImplementedError(
            "Custom SDPA implementation is not supported for CUDA yet. Please use 'flash_attention' instead."
        )

    # print("Autotuning...")
    # os.environ["TRITON_PRINT_AUTOTUNING"] = "1"
    # autotuning_inputs = torch.rand(
    #     1, 128, 3000, dtype=torch.bfloat16, device=torch.device("cuda")
    # )

    # decoder_input_embeds = torch.rand(
    #     1, 1500, 1280, dtype=torch.bfloat16, device=torch.device("cuda")
    # )

    # with torch.no_grad():
    #     for _ in range(5):
    #         model.model(autotuning_inputs, decoder_inputs_embeds=decoder_input_embeds)

    # print("Done.")

    # Decomposes SDPA since we don't have a flash attention kernel for it yet.
    # with torch.nn.attention.sdpa_kernel([SDPBackend.MATH]), torch.no_grad():
    exported_progs = model.export()

    ret: dict[str, ExecutorchProgram] = _lower_to_executorch(
        exported_progs, model.metadata
    )

    return ret
