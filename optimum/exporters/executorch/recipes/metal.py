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
import math
from typing import Dict, Union

import torch
from packaging.version import parse

from executorch import version as executorch_version


EXECUTORCH_VERSION = parse(executorch_version.__version__)
METAL_BACKEND_AVAILABLE = EXECUTORCH_VERSION >= parse("1.1.0.dev20251017")

METAL_SUPPORTED_HEAD_DIMS = (64, 96, 128)

if METAL_BACKEND_AVAILABLE:
    try:
        from executorch.backends.apple.metal.metal_backend import MetalBackend
        from executorch.backends.apple.metal.metal_partitioner import MetalPartitioner
    except ImportError:
        METAL_BACKEND_AVAILABLE = False


def _sdpa_decomposition(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False):
    """Decompose scaled_dot_product_attention into matmul + softmax.

    The Metal SDPA kernel only supports head_dim in {64, 96, 128}.
    For models with other head dimensions (e.g. Gemma3 with head_dim=256),
    we decompose into ops that AOTI can compile for Metal: matmul, softmax, etc.
    """
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_weight = torch.ops.aten.matmul.default(
        query, torch.ops.aten.transpose.int(key, -2, -1)
    )
    attn_weight = torch.ops.aten.mul.Scalar(attn_weight, scale_factor)
    if is_causal:
        causal_mask = torch.ones(L, S, dtype=torch.bool, device=query.device).tril()
        attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
        attn_bias = torch.ops.aten.masked_fill.Scalar(attn_bias, ~causal_mask, float("-inf"))
        attn_weight = torch.ops.aten.add.Tensor(attn_weight, attn_bias)
    if attn_mask is not None:
        attn_weight = torch.ops.aten.add.Tensor(attn_weight, attn_mask)
    attn_weight = torch.ops.aten.softmax.int(attn_weight, -1)
    return torch.ops.aten.matmul.default(attn_weight, value)


def _linear_bias_decomposition(input, weight, bias=None):
    """Decompose linear with bias into matmul + add.

    Avoids Metal backend issues with reinterpret_tensor_wrapper when
    linear layers have biases (0-stride problem).
    """
    weight_t = torch.ops.aten.t.default(weight)
    out = torch.ops.aten.matmul.default(input, weight_t)
    if bias is not None:
        return torch.ops.aten.add.Tensor(out, bias)
    return out


if METAL_BACKEND_AVAILABLE:
    from tabulate import tabulate
    from torch.export import ExportedProgram

    from executorch.backends.apple.metal.metal_backend import MetalBackend
    from executorch.backends.apple.metal.metal_partitioner import MetalPartitioner
    from executorch.devtools.backend_debug import get_delegation_info
    from executorch.exir import (
        EdgeCompileConfig,
        ExecutorchProgram,
        to_edge_transform_and_lower,
    )
    from optimum.executorch.passes.remove_padding_idx_embedding_pass import RemovePaddingIdxEmbeddingPass

    from ..integrations import (
        CausalLMExportableModule,
        MaskedLMExportableModule,
        MultiModalTextToTextExportableModule,
        Seq2SeqLMExportableModule,
    )
    from ..recipe_registry import register_recipe

    @register_recipe("metal")
    def export_to_executorch_with_metal(
        model: Union[
            CausalLMExportableModule,
            MaskedLMExportableModule,
            Seq2SeqLMExportableModule,
            MultiModalTextToTextExportableModule,
        ],
        **kwargs,
    ):
        """
        Export a PyTorch model to ExecuTorch w/ delegation to Metal backend.

        This function also write metadata required by the ExecuTorch runtime to the model.

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

        def _lower_to_executorch(
            exported_programs: Dict[str, ExportedProgram],
            metadata=None,
        ) -> Dict[str, ExecutorchProgram]:
            logging.debug(f"\nExported program: {exported_programs}")

            # If just one exported program, the method name in the .pte for it should be "forward".
            if len(exported_programs) == 1:
                exported_programs = {"forward": next(iter(exported_programs.values()))}

            partitioners = {
                key: [MetalPartitioner([MetalBackend.generate_method_name_compile_spec(key)])]
                for key in exported_programs.keys()
            }

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
                logging.debug(f"---------------------- Method: {method} ----------------------")
                logging.debug(
                    f"\nExecuTorch program for {pte_name}.pte: {et_prog.exported_program(method).graph_module}"
                )
                delegation_info = get_delegation_info(et_prog.exported_program(method).graph_module)
                logging.debug(f"\nDelegation info Summary for {pte_name}.pte: {delegation_info.get_summary()}")
                logging.debug(
                    f"\nDelegation info for {pte_name}.pte: {tabulate(delegation_info.get_operator_delegation_dataframe(), headers='keys', tablefmt='fancy_grid')}"
                )
            return {pte_name: et_prog}

        if (
            model.config._attn_implementation == "custom_sdpa"
            or model.config._attn_implementation == "custom_sdpa_ring_kv_cache"
        ):
            raise NotImplementedError("Custom SDPA implementation is not supported for Metal.")

        # Metal uses standard SDPA, not custom SDPA with custom KV cache.
        if hasattr(model, "use_custom_sdpa"):
            model.use_custom_sdpa = False
        if hasattr(model, "use_custom_kv_cache"):
            model.use_custom_kv_cache = False

        exported_progs = model.export()

        # Decompose ops that the Metal backend cannot handle natively.
        decomp_table = {
            torch.ops.aten.linear.default: _linear_bias_decomposition,
        }

        # The Metal SDPA kernel only supports head_dim in {64, 96, 128}.
        # For models with unsupported head_dim, decompose SDPA into matmul + softmax.
        head_dim = getattr(model.config, "head_dim", None)
        if head_dim is None:
            text_config = getattr(model.config, "text_config", None)
            if text_config is not None:
                head_dim = getattr(text_config, "head_dim", None)

        if head_dim is not None and head_dim not in METAL_SUPPORTED_HEAD_DIMS:
            logging.info(
                f"Model head_dim={head_dim} is not natively supported by Metal SDPA kernel "
                f"(supported: {METAL_SUPPORTED_HEAD_DIMS}). Decomposing SDPA into matmul + softmax."
            )
            decomp_table[torch.ops.aten.scaled_dot_product_attention.default] = _sdpa_decomposition

        exported_progs = {
            key: ep.run_decompositions(decomp_table)
            for key, ep in exported_progs.items()
        }

        return _lower_to_executorch(exported_progs, model.metadata)
