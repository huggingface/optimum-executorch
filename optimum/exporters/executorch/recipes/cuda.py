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
from typing import Dict, Union

import torch
from tabulate import tabulate
from torch.export import ExportedProgram

from executorch.devtools.backend_debug import get_delegation_info
from executorch.exir import (
    EdgeCompileConfig,
    ExecutorchBackendConfig,
    ExecutorchProgram,
    to_edge_transform_and_lower,
)
from executorch.exir.backend.compile_spec_schema import CompileSpec
from executorch.exir.passes import MemoryPlanningPass
from optimum.executorch.passes.remove_padding_idx_embedding_pass import (
    RemovePaddingIdxEmbeddingPass,
)

from ..integrations import (
    CausalLMExportableModule,
    MaskedLMExportableModule,
    MultiModalTextToTextExportableModule,
    Seq2SeqLMExportableModule,
)
from ..recipe_registry import register_recipe


aten = torch.ops.aten


def lower_to_executorch(
    exported_programs: Dict[str, ExportedProgram],
    metadata=None,
    is_windows: bool = False,
    model_config=None,
) -> Dict[str, ExecutorchProgram]:
    # Import here to avoid version conflicts.
    from torch._inductor.decomposition import conv1d_to_conv2d

    from executorch.backends.cuda.cuda_backend import CudaBackend
    from executorch.backends.cuda.cuda_partitioner import CudaPartitioner

    logging.debug(f"\nExported program: {exported_programs}")

    # If just one exported program, the method name in the .pte for it should be "forward".
    if len(exported_programs) == 1:
        exported_programs = {"forward": next(iter(exported_programs.values()))}

    # Check if this is a Gemma3 model
    model_type = getattr(model_config, "model_type", None) if model_config else None

    # CUDA backend compile spec with method name.
    partitioners = {}
    for key in exported_programs.keys():
        compile_specs = [CudaBackend.generate_method_name_compile_spec(key)]
        if is_windows:
            compile_specs.append(CompileSpec("platform", "windows".encode("utf-8")))

        # Add Gemma3-specific compile spec if needed
        if model_type == "gemma3":
            compile_specs.append(CompileSpec(key="triton_kernel_mode", value=b"OFF"))

        partitioners[key] = [CudaPartitioner(compile_specs)]

    # Add decompositions for triton to generate kernels.
    for key, ep in exported_programs.items():
        exported_programs[key] = ep.run_decompositions(
            {
                aten.conv1d.default: conv1d_to_conv2d,
            }
        )
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
    et_prog = et_prog.to_executorch(
        ExecutorchBackendConfig(
            memory_planning_pass=MemoryPlanningPass(
                alloc_graph_input=False,
            )
        ),
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
    if (
        model.config._attn_implementation == "custom_sdpa"
        or model.config._attn_implementation == "custom_sdpa_ring_kv_cache"
    ):
        raise NotImplementedError(
            "Custom SDPA implementation is not supported for CUDA yet. Please use 'flash_attention' instead."
        )

    exported_progs = model.export()

    return lower_to_executorch(exported_progs, model.metadata, model_config=getattr(model, "config", None))
