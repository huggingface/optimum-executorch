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

from typing import Dict, Union

from torch.export import ExportedProgram

from executorch.exir import (
    EdgeCompileConfig,
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


@register_recipe("portable")
def export_to_executorch_with_portable(
    model: Union[CausalLMExportableModule, MaskedLMExportableModule, Seq2SeqLMExportableModule],
    **kwargs,
):
    """
    Export a PyTorch model to ExecuTorch with Portable kernels.

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

    def _lower_to_executorch(
        exported_programs: Dict[str, ExportedProgram],
        metadata=None,
    ) -> Dict[str, ExecutorchProgram]:
        # If just one exported program, the method name in the .pte for it should be "forward".
        if len(exported_programs) == 1:
            exported_programs = {"forward": next(iter(exported_programs.values()))}

        et_prog = to_edge_transform_and_lower(
            exported_programs,
            partitioner=[],
            compile_config=EdgeCompileConfig(
                _check_ir_validity=False,
                _skip_dim_order=True,
            ),
            constant_methods=metadata,
            transform_passes=[RemovePaddingIdxEmbeddingPass()],
        ).to_executorch()
        pte_name = "model"
        return {pte_name: et_prog}

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
