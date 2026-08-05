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

import functools
import logging
from typing import Dict, Union

import torch
from packaging.version import parse
from tabulate import tabulate
from torch import __version__ as torch_version
from torch.export import ExportedProgram
from torchao.utils import unwrap_tensor_subclass

from executorch import version as executorch_version
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.devtools.backend_debug import get_delegation_info
from executorch.exir import (
    EdgeCompileConfig,
    ExecutorchBackendConfig,
    ExecutorchProgram,
    to_edge_transform_and_lower,
)
from executorch.exir.passes import MemoryPlanningPass
from optimum.executorch.passes.remove_padding_idx_embedding_pass import RemovePaddingIdxEmbeddingPass

from ..integrations import (
    CausalLMExportableModule,
    MaskedLMExportableModule,
    MultiModalTextToTextExportableModule,
    Seq2SeqLMExportableModule,
)
from ..recipe_registry import register_recipe


# First ExecuTorch nightly whose XNNPACK partitioner understands `enable_bf16`.
_MIN_ET_FOR_BF16_DELEGATION = "1.4.0.dev20260801"


@functools.lru_cache(maxsize=1)
def _xnnpack_honors_enable_bf16() -> bool:
    """
    Whether the installed XNNPACK partitioner actually acts on `enable_bf16`.

    The flag is forwarded through `**kwargs` into every partitioner config, so an ExecuTorch that
    predates it accepts `enable_bf16=True` and silently ignores it. Probe the behavior rather than
    comparing versions: source builds report versions such as `1.4.0a0+<sha>`, which PEP 440 sorts
    *after* every `1.4.0.devN` nightly and would pass a version check whether or not they carry the
    feature.
    """
    try:
        configs = XnnpackPartitioner(enable_bf16=True).target_partitioner_configs.values()
    except Exception:  # Partitioner internals differ across versions; assume unsupported.
        return False
    return any(getattr(config, "enable_bf16", False) for config in configs)


@register_recipe("xnnpack")
def export_to_executorch_with_xnnpack(
    model: Union[
        CausalLMExportableModule,
        MaskedLMExportableModule,
        Seq2SeqLMExportableModule,
        MultiModalTextToTextExportableModule,
    ],
    **kwargs,
):
    """
    Export a PyTorch model to ExecuTorch w/ delegation to XNNPACK backend.

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
        backend_config_dict = {
            "extract_delegate_segments": True,
            "memory_planning_pass": MemoryPlanningPass(alloc_graph_input=False),
        }
        backend_config_dict["do_quant_fusion_and_const_prop"] = True
        logging.debug(f"\nExported program: {exported_programs}")

        # If just one exported program, the method name in the .pte for it should be "forward".
        if len(exported_programs) == 1:
            exported_programs = {"forward": next(iter(exported_programs.values()))}

        # bf16 delegation is opt-in in the backend. A bf16 model always carries bf16 tensors
        # (e.g. RMSNorm weights), even when its linears are quantized.
        enable_bf16 = any(
            getattr(tensor, "dtype", None) == torch.bfloat16
            for exported_program in exported_programs.values()
            for tensor in exported_program.state_dict.values()
        )
        if enable_bf16 and not _xnnpack_honors_enable_bf16():
            reason = (
                f"the installed ExecuTorch ({executorch_version.__version__}) cannot delegate bf16 to XNNPACK "
                f"(that needs ExecuTorch >= {_MIN_ET_FOR_BF16_DELEGATION})"
            )
            # ExecuTorch has no portable kernels for these, so leaving them undelegated makes
            # `to_executorch` die later with a far less obvious "Missing out variants: torchao::...".
            has_affine_quant_ops = any(
                node.op == "call_function" and "torchao" in str(node.target) and "affine" in str(node.target)
                for exported_program in exported_programs.values()
                for node in exported_program.graph.nodes
            )
            if has_affine_quant_ops:
                raise RuntimeError(
                    "Quantized linears (--qlinear) lower only through XNNPACK delegation, since torchao's "
                    f"affine quant ops have no portable ExecuTorch kernels, but {reason}. "
                    "Upgrade ExecuTorch, or re-export with --dtype float32 or --dtype float16."
                )
            logging.warning(
                f"Exporting a bf16 model but {reason}, so every bf16 operator will fall back to the portable "
                "kernels and inference will be slow. Upgrade ExecuTorch to delegate bf16 to XNNPACK."
            )

        et_prog = to_edge_transform_and_lower(
            exported_programs,
            partitioner=[XnnpackPartitioner(enable_bf16=enable_bf16)],
            compile_config=EdgeCompileConfig(
                _check_ir_validity=False,
                _skip_dim_order=True,
            ),
            constant_methods=metadata,
            transform_passes=[RemovePaddingIdxEmbeddingPass()],
        )
        et_prog = et_prog.to_executorch(
            config=ExecutorchBackendConfig(**backend_config_dict),
        )
        pte_name = "model"
        for method in et_prog.methods:
            logging.debug(f"---------------------- Method: {method} ----------------------")
            logging.debug(f"\nExecuTorch program for {pte_name}.pte: {et_prog.exported_program(method).graph_module}")
            delegation_info = get_delegation_info(et_prog.exported_program(method).graph_module)
            logging.debug(f"\nDelegation info Summary for {pte_name}.pte: {delegation_info.get_summary()}")
            logging.debug(
                f"\nDelegation info for {pte_name}.pte: {tabulate(delegation_info.get_operator_delegation_dataframe(), headers='keys', tablefmt='fancy_grid')}"
            )
        return {pte_name: et_prog}

    # TODO: remove after ExecuTorch dep on Torch >= 2.10.0.
    if parse(torch_version) < parse("2.10.0.dev20251104"):
        model = unwrap_tensor_subclass(model)
    exported_progs = model.export()

    if (
        model.config._attn_implementation == "custom_sdpa"
        or model.config._attn_implementation == "custom_sdpa_ring_kv_cache"
    ):
        # Sanity check to make sure the exported program contains the custom sdpa operator.
        if not any(
            node.op == "call_function" and "custom_sdpa" in str(node.target)
            for exported_program in exported_progs.values()
            for node in exported_program.graph_module.graph.nodes
        ):
            raise ValueError("'custom_sdpa' not found in the graph.")

    return _lower_to_executorch(exported_progs, model.metadata)
