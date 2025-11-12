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

"""Entry point to the optimum.exporters.executorch command line."""

import argparse
import logging
import os
import warnings
from pathlib import Path

from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE

from optimum.utils.import_utils import is_transformers_version
from transformers import PretrainedConfig
from transformers.utils import is_torch_available

from ...commands.export.executorch import parse_args_executorch
from .convert import export_to_executorch
from .task_registry import discover_tasks, task_registry


if is_torch_available():
    pass

import math
from typing import Any, Optional, Union

import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton


@triton.autotune(
    configs=[
        # Favor configs tuned for HEAD_DIM=64 and L up to ~1500
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_stages=4, num_warps=8),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256}, num_stages=4, num_warps=8),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 256}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_stages=3, num_warps=4),
    ],
    key=["L", "HEAD_DIM"],
)
@triton.jit
def _sdpa_fwd_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    B,
    H,
    L,
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
    stride_ob,
    stride_oh,
    stride_ol,
    stride_od,
    sm_scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM_CE: tl.constexpr,
):
    # Program IDs
    pid_m = tl.program_id(axis=0)  # along query length
    pid_hz = tl.program_id(axis=1)  # flattened batch*head

    off_b = pid_hz // H
    off_h = pid_hz % H

    # Compute ranges
    start_m = pid_m * BLOCK_M
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM_CE)
    mask_m = offs_m < L

    # Base pointers for this (b, h)
    q_base = q_ptr + off_b * stride_qb + off_h * stride_qh
    k_base = k_ptr + off_b * stride_kb + off_h * stride_kh
    v_base = v_ptr + off_b * stride_vb + off_h * stride_vh
    o_base = o_ptr + off_b * stride_ob + off_h * stride_oh

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

    # Loop over keys/values along sequence length in tiles of BLOCK_N
    # Load K as [BLOCK_N, HEAD_DIM] for coalesced reads, then use tl.trans(K) in dot
    for start_n in tl.range(0, L, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < L

        # Load K tile [BLOCK_N, HEAD_DIM] (contiguous along HEAD_DIM)
        k_ptrs = k_base + (
            offs_n[:, None] * stride_kl + offs_d_ctg[None, :] * stride_kd
        )
        k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
        k = k.to(tl.bfloat16)

        # Compute attention logits [BLOCK_M, BLOCK_N] = Q[BM,D] @ K[BN,D]^T
        qk = tl.dot(q, tl.trans(k)).to(tl.float32)  # accumulator in fp32
        qk = qk * qk_scale

        # Apply OOB masks for both rows and cols to keep stability
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
        # Cast p to bf16 to use tensor-cores in tl.dot; accumulate in fp32
        p_bf16 = p.to(tl.bfloat16)
        acc = tl.dot(p_bf16, v, acc)

        # Update softmax stats
        l_i = l_i * alpha + l_ij
        m_i = m_ij

    # Normalize accumulator by softmax denominator
    acc = acc / l_i[:, None]

    # Store output [BLOCK_M, HEAD_DIM]
    o_ptrs = o_base + (offs_m[:, None] * stride_ol + offs_d_ctg[None, :] * stride_od)
    tl.store(o_ptrs, acc.to(tl.bfloat16), mask=mask_m[:, None])


@triton_op("custom::optimized_triton_scaled_dot_product_attention", mutates_args={})
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
    Triton fused Scaled Dot-Product Attention (forward, no causal, no dropout).
    Expected shapes (tested): [B=1, H=20, L<=1500, D=64], dtype bfloat16.

    Args:
        query: Query tensor [B, H, L, D]
        key: Key tensor [B, H, L, D]
        value: Value tensor [B, H, L, D]
        attn_mask: must be None (not supported)
        dropout_p: must be 0.0 (not supported)
        is_causal: must be False (not supported)
        scale: must be 0.0 (not supported)
        enable_gqa: must be False (not supported)

    Returns:
        Output tensor [B, H, L, D]
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
    if query.shape != key.shape or query.shape != value.shape:
        raise RuntimeError(
            f"Q, K, V must have identical shapes; got query={query.shape}, key={key.shape}, value={value.shape}."
        )
    if query.dim() != 4:
        raise RuntimeError(
            f"Expected 4D tensors shaped [B, H, L, D]; got {query.dim()}D."
        )

    # Enforce that only default values are accepted for these arguments
    if attn_mask is not None:
        raise RuntimeError(
            "attn_mask must be None (not supported in this implementation)."
        )
    if dropout_p != 0.0:
        raise RuntimeError(
            "dropout_p must be 0.0 (not supported in this implementation)."
        )
    if is_causal is not False:
        raise RuntimeError(
            "is_causal must be False (not supported in this implementation)."
        )
    if scale != 0.0:
        raise RuntimeError("scale must be 0.0 (not supported in this implementation).")
    if enable_gqa is not False:
        raise RuntimeError(
            "enable_gqa must be False (not supported in this implementation)."
        )

    B, H, L, D = query.shape
    # Allocate output
    out = torch.empty_like(query)

    # Element-wise strides (in elements)
    sqb, sqh, sql, sqd = query.stride()
    skb, skh, skl, skd = key.stride()
    svb, svh, svl, svd = value.stride()
    sob, soh, sol, sod = out.stride()

    # Grid: tile queries (M) and batch*heads axis
    def grid(META):
        return (
            triton.cdiv(L, META["BLOCK_M"]),
            B * H,
        )

    # Scale factor for SDPA
    sm_scale = 1.0 / math.sqrt(D)

    # Launch kernel using wrap_triton to avoid tracing issues during export/compile
    # Note: wrap_triton returns a callable that can be indexed with grid
    wrap_triton(_sdpa_fwd_kernel)[grid](
        query,
        key,
        value,
        out,
        B,
        H,
        L,
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
        sob,
        soh,
        sol,
        sod,
        sm_scale,
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
    """
    # Validate shapes match
    assert query.shape == key.shape == value.shape, "Q, K, V must have the same shape"
    assert query.dtype == key.dtype == value.dtype, "Q, K, V must have the same dtype"

    # Output has the same shape and dtype as query
    # IMPORTANT: Use the exact same dtype to satisfy ExecuTorch validation
    return torch.empty_like(query, dtype=query.dtype, device=query.device)

torch.nn.functional.scaled_dot_product_attention = (
    optimized_triton_scaled_dot_product_attention
)

def main_export(
    model_name_or_path: str,
    task: str,
    recipe: str,
    output_dir: Union[str, Path],
    config: Optional[PretrainedConfig] = None,
    cache_dir: str = HUGGINGFACE_HUB_CACHE,
    trust_remote_code: bool = False,
    pad_token_id: Optional[int] = None,
    subfolder: str = "",
    revision: str = "main",
    force_download: bool = False,
    local_files_only: bool = False,
    use_auth_token: Optional[Union[bool, str]] = None,
    token: Optional[Union[bool, str]] = None,
    **kwargs,
):
    """
    Full-suite ExecuTorch export function, exporting **from a model ID on Hugging Face Hub or a local model repository**.

    Args:
        model_name_or_path (`str`):
            Model ID on huggingface.co or path on disk to the model repository to export. Example: `model_name_or_path="meta-llama/Llama-3.2-1B"` or `mode_name_or_path="/path/to/model_folder`.
        task (`str`):
            The task to export the model for, e.g. "text-generation".
        recipe (`str`):
            The recipe to use to do the export, e.g. "xnnpack".
        output_dir (`Union[str, Path]`):
            Path indicating the directory where to store the generated ExecuTorch model.
        config (`Optional[PretrainedConfig]`, defaults to `None`):
            The model configuration to use to load the model.
        cache_dir (`Optional[str]`, defaults to `None`):
            Path indicating where to store cache. The default Hugging Face cache path will be used by default.
        trust_remote_code (`bool`, defaults to `False`):
            Allows to use custom code for the modeling hosted in the model repository. This option should only be set for repositories
            you trust and in which you have read the code, as it will execute on your local machine arbitrary code present in the
            model repository.
        pad_token_id (`Optional[int]`, defaults to `None`):
            This is needed by some models, for some tasks. If not provided, will attempt to use the tokenizer to guess it.
        subfolder (`str`, defaults to `""`):
            In case the relevant files are located inside a subfolder of the model repo either locally or on huggingface.co, you can
            specify the folder name here.
        revision (`str`, defaults to `"main"`):
            Revision is the specific model version to use. It can be a branch name, a tag name, or a commit id.
        force_download (`bool`, defaults to `False`):
            Whether or not to force the (re-)download of the model weights and configuration files, overriding the
            cached versions if they exist.
        local_files_only (`Optional[bool]`, defaults to `False`):
            Whether or not to only look at local files (i.e., do not try to download the model).
        use_auth_token (`Optional[Union[bool,str]]`, defaults to `None`):
            Deprecated. Please use the `token` argument instead.
        token (`Optional[Union[bool,str]]`, defaults to `None`):
            The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
            when running `huggingface-cli login` (stored in `huggingface_hub.constants.HF_TOKEN_PATH`).
        **kwargs:
            Additional configuration options to tasks and recipes.

    Example usage:
    ```python
    >>> from optimum.exporters.executorch import main_export

    >>> main_export("meta-llama/Llama-3.2-1B", "text-generation", "xnnpack", "meta_llama3_2_1b/")
    ```
    """
    if is_transformers_version("<", "4.46"):
        raise ValueError(
            "The minimum Transformers version compatible with ExecuTorch is 4.46.0. Please upgrade to Transformers 4.46.0 or later."
        )

    if use_auth_token is not None:
        warnings.warn(
            "The `use_auth_token` argument is deprecated and will be removed soon. Please use the `token` argument instead.",
            FutureWarning,
        )
        if token is not None:
            raise ValueError(
                "You cannot use both `use_auth_token` and `token` arguments at the same time."
            )
        token = use_auth_token

    # Dynamically discover and import registered tasks
    discover_tasks()

    # Load the model for specific task
    try:
        task_func = task_registry.get(task)
    except KeyError as e:
        raise RuntimeError(f"The task '{task}' isn't registered. Detailed error: {e}")

    kwargs["cache_dir"] = cache_dir
    kwargs["trust_remote_code"] = trust_remote_code
    kwargs["subfolder"] = subfolder
    kwargs["revision"] = revision
    kwargs["force_download"] = force_download
    kwargs["config"] = config

    # 1. Load model, apply source transformations, and torch.export() into a graph (ExportedProgram).
    logging.info(f"Loading {model_name_or_path} and exporting to static graph...")
    recipe_kwargs = kwargs.pop("recipe_kwargs", {})

    model = task_func(model_name_or_path, **kwargs)

    # 2. Export to ExecuTorch through ExecuTorch's lowering APIs.
    logging.info(f"Lowering {model_name_or_path} to ExecuTorch...")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    return export_to_executorch(
        model=model,
        task=task,
        recipe=recipe,
        output_dir=output_dir,
        **recipe_kwargs,
    )


def main():
    parser = argparse.ArgumentParser("Hugging Face Optimum ExecuTorch exporter")

    parse_args_executorch(parser)

    # Retrieve CLI arguments
    args = parser.parse_args()

    main_export(
        model_name_or_path=args.model,
        output_dir=args.output_dir,
        task=args.task,
        recipe=args.recipe,
        cache_dir=args.cache_dir,
        trust_remote_code=args.trust_remote_code,
        pad_token_id=args.pad_token_id,
    )


if __name__ == "__main__":
    main()
