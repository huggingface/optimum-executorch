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
from itertools import product
from typing import Any, Dict, Union

from tabulate import tabulate
from torch.export import ExportedProgram

from executorch.devtools.backend_debug import get_delegation_info
from executorch.exir import (
    EdgeCompileConfig,
    ExecutorchBackendConfig,
    ExecutorchProgram,
    to_edge_transform_and_lower,
)
from optimum.executorch.passes.remove_padding_idx_embedding_pass import RemovePaddingIdxEmbeddingPass

from ..integrations import (
    CausalLMExportableModule,
    MaskedLMExportableModule,
    Seq2SeqLMExportableModule,
)
from ..recipe_registry import register_recipe


@register_recipe("vulkan")
def _export_to_executorch(
    model: Union[CausalLMExportableModule, MaskedLMExportableModule, Seq2SeqLMExportableModule],
    **kwargs,
):
    """
    Export a PyTorch model to ExecuTorch w/ delegation to CoreML backend.

    This function also write metadata required by the ExecuTorch runtime to the model.

    Args:
        model (Union[CausalLMExportableModule, MaskedLMExportableModule, Seq2SeqLMExportableModule]):
            The PyTorch model to be exported to ExecuTorch.
        **kwargs:
            Additional keyword arguments for recipe-specific configurations, e.g. export using different example inputs, or different compile/bechend configs.

    Returns:
        Dict[str, ExecutorchProgram]:
            A map of exported and optimized program for ExecuTorch.
            For encoder-decoder models or multimodal models, it may generate multiple programs.
    """
    # Import here because coremltools might not be available in all environments
    from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner

    def _lower_to_executorch(
        exported_programs: Dict[str, ExportedProgram],
        metadata=None,
    ) -> Dict[str, ExecutorchProgram]:
        logging.debug(f"\nExported program: {exported_programs}")

        # If just one exported program, the method name in the .pte for it should be "forward".
        if len(exported_programs) == 1:
            exported_programs = {"forward": next(iter(exported_programs.values()))}

        et_prog = to_edge_transform_and_lower(
            exported_programs,
            partitioner=[
                VulkanPartitioner(
                    {"require_dynamic_shapes": True, "force_fp16": True}
                )
            ],
            compile_config=EdgeCompileConfig(
                _check_ir_validity=False,
                # In ET 0.7, we can set _skip_dim_order=False
                _skip_dim_order=True,
            ),
            constant_methods=metadata,
            transform_passes=[RemovePaddingIdxEmbeddingPass()],
        ).to_executorch()
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

    exported_progs = model.export()
    return _lower_to_executorch(exported_progs, model.metadata)
