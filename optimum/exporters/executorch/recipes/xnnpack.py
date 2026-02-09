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

import copy
import logging
import operator
from typing import Dict, Union

import torch
from packaging.version import parse
from tabulate import tabulate
from torch import __version__ as torch_version
from torch.export import ExportedProgram
from torch.export.graph_signature import (
    ExportGraphSignature,
    OutputKind,
    OutputSpec,
    TensorArgument,
)
from torchao.utils import unwrap_tensor_subclass

from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.devtools.backend_debug import get_delegation_info
from executorch.exir import (
    EdgeCompileConfig,
    ExecutorchBackendConfig,
    ExecutorchProgram,
    to_edge_transform_and_lower,
)
from executorch.exir.passes.init_mutable_pass import InitializedMutableBufferPass
from executorch.exir.passes import MemoryPlanningPass
from optimum.executorch.passes.remove_padding_idx_embedding_pass import RemovePaddingIdxEmbeddingPass

from ..integrations import (
    CausalLMExportableModule,
    MaskedLMExportableModule,
    MultiModalTextToTextExportableModule,
    Seq2SeqLMExportableModule,
)
from ..recipe_registry import register_recipe


def _maybe_add_cross_cache_mutations(
    exported_program: ExportedProgram,
) -> ExportedProgram:
    """
    WhisperCrossAttention uses torch.cond for cross-attention cache reuse/update.
    In some torch versions, mutation edges for these caches are not surfaced in
    the exported graph signature, so they may be planned as read-only constants.
    Add explicit BUFFER_MUTATION outputs for cross-attention cache tensors by
    wiring cond getitem outputs into graph outputs/signature.
    """

    graph_signature = exported_program.graph_signature
    inputs_to_buffers = graph_signature.inputs_to_buffers
    if not inputs_to_buffers:
        return exported_program

    existing_mutation_targets = {
        output_spec.target
        for output_spec in graph_signature.output_specs
        if output_spec.kind == OutputKind.BUFFER_MUTATION and output_spec.target is not None
    }

    graph_module = copy.deepcopy(exported_program.graph_module)
    mutation_candidates = []
    for node in graph_module.graph.nodes:
        if node.op != "call_function" or node.target != torch.ops.higher_order.cond:
            continue

        if len(node.args) < 4:
            continue
        operands = node.args[3]
        if not isinstance(operands, (tuple, list)) or len(operands) < 2:
            continue

        key_operand, value_operand = operands[0], operands[1]
        if not isinstance(key_operand, torch.fx.Node) or not isinstance(
            value_operand, torch.fx.Node
        ):
            continue

        key_cache_name = inputs_to_buffers.get(key_operand.name)
        value_cache_name = inputs_to_buffers.get(value_operand.name)
        if key_cache_name is None or value_cache_name is None:
            continue
        if not key_cache_name.startswith("cross_attention_key_cache_"):
            continue
        if not value_cache_name.startswith("cross_attention_value_cache_"):
            continue

        key_output = None
        value_output = None
        for user in node.users:
            if user.op != "call_function" or user.target != operator.getitem:
                continue
            idx = user.args[1]
            if idx == 0:
                key_output = user
            elif idx == 1:
                value_output = user

        if key_output is None or value_output is None:
            continue
        mutation_candidates.append((key_cache_name, key_output))
        mutation_candidates.append((value_cache_name, value_output))

    if not mutation_candidates:
        return exported_program

    deduped_nodes = {}
    for buffer_name, node in mutation_candidates:
        deduped_nodes.setdefault(buffer_name, node)

    def _cache_sort_key(buffer_name: str):
        cache_idx = int(buffer_name.rsplit("_", 1)[1])
        key_or_value = 0 if "_key_cache_" in buffer_name else 1
        return cache_idx, key_or_value

    additions = [
        (buffer_name, deduped_nodes[buffer_name])
        for buffer_name in sorted(deduped_nodes, key=_cache_sort_key)
        if buffer_name not in existing_mutation_targets
    ]
    if not additions:
        return exported_program

    output_specs = list(graph_signature.output_specs)
    first_user_output_idx = next(
        (
            idx
            for idx, output_spec in enumerate(output_specs)
            if output_spec.kind != OutputKind.BUFFER_MUTATION
        ),
        len(output_specs),
    )

    output_node = graph_module.graph.output_node()
    graph_outputs = list(output_node.args[0])
    graph_outputs = (
        graph_outputs[:first_user_output_idx]
        + [node for _, node in additions]
        + graph_outputs[first_user_output_idx:]
    )
    output_node.args = (tuple(graph_outputs),)
    graph_module.graph.lint()
    graph_module.recompile()

    new_output_specs = output_specs[:first_user_output_idx]
    for buffer_name, node in additions:
        new_output_specs.append(
            OutputSpec(
                kind=OutputKind.BUFFER_MUTATION,
                arg=TensorArgument(name=node.name),
                target=buffer_name,
            )
        )
    new_output_specs.extend(output_specs[first_user_output_idx:])

    new_graph_signature = ExportGraphSignature(
        input_specs=list(graph_signature.input_specs),
        output_specs=new_output_specs,
    )

    patched_program = ExportedProgram(
        root=graph_module,
        graph=graph_module.graph,
        graph_signature=new_graph_signature,
        state_dict=exported_program.state_dict,
        range_constraints=exported_program.range_constraints,
        module_call_graph=copy.deepcopy(exported_program.module_call_graph),
        example_inputs=exported_program.example_inputs,
        constants=exported_program.constants,
        verifiers=[exported_program.verifier],
    )
    patched_program.graph_module.meta.update(exported_program.graph_module.meta)
    logging.info(
        "Added %d cross-attention cache mutation edges for ExecuTorch export.",
        len(additions),
    )
    return patched_program


def _patch_cross_cache_mutations_for_exported_programs(
    exported_programs: Dict[str, ExportedProgram],
) -> Dict[str, ExportedProgram]:
    return {
        name: _maybe_add_cross_cache_mutations(program)
        for name, program in exported_programs.items()
    }


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
            "passes": [
                # Preserve initial values for newly-mutable Whisper cross-attn
                # caches and cache-initialized flags.
                InitializedMutableBufferPass(
                    [
                        "cross_attention_key_cache_",
                        "cross_attention_value_cache_",
                        "encoder_attn_cache_initialized",
                    ]
                )
            ],
        }
        backend_config_dict["do_quant_fusion_and_const_prop"] = True
        logging.debug(f"\nExported program: {exported_programs}")

        # If just one exported program, the method name in the .pte for it should be "forward".
        if len(exported_programs) == 1:
            exported_programs = {"forward": next(iter(exported_programs.values()))}

        exported_programs = _patch_cross_cache_mutations_for_exported_programs(
            exported_programs
        )

        et_prog = to_edge_transform_and_lower(
            exported_programs,
            partitioner=[XnnpackPartitioner()],
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
