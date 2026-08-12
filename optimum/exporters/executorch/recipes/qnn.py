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

from executorch.backends.qualcomm._passes.qnn_pass_manager import QnnPassManager
from executorch.backends.qualcomm.partition.qnn_partitioner import (
    generate_qnn_executorch_option,
    get_skip_decomp_table,
    QnnPartitioner,
)

# Import here because QNNtools might not be available in all environments
from executorch.backends.qualcomm.qnn_preprocess import QnnBackend
from executorch.backends.qualcomm.quantizer.quantizer import QnnQuantizer
from executorch.backends.qualcomm.utils.utils import (
    generate_htp_compiler_spec,
    generate_qnn_executorch_compiler_spec,
    get_soc_to_chipset_map,
    qnn_edge_config,
    to_edge_transform_and_lower_to_qnn,
)

from executorch.devtools.backend_debug import get_delegation_info
from executorch.exir import (
    EdgeCompileConfig,
    ExecutorchBackendConfig,
    ExecutorchProgram,
    to_edge_transform_and_lower,
)

from tabulate import tabulate
from torch.export import ExportedProgram

from ..integrations import (
    CausalLMExportableModule,
    MaskedLMExportableModule,
    Seq2SeqLMExportableModule,
)
from ..recipe_registry import register_recipe


def _export_to_executorch(
    model: Union[
        CausalLMExportableModule, MaskedLMExportableModule, Seq2SeqLMExportableModule
    ],
    **kwargs,
):
    """
    Export a PyTorch model to ExecuTorch w/ delegation to QNN backend.

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
        metadata,
        dtype,
        soc,
    ) -> Dict[str, ExecutorchProgram]:

        et_progs = {}
        backend_config_dict = {}
        compiler_spec = generate_qnn_executorch_compiler_spec(
            soc_model=get_soc_to_chipset_map()[soc],
            backend_options=generate_htp_compiler_spec(use_fp16=True),
        )
        aten_programs = {}
        transform_passes = {}
        qnn_partitioner = QnnPartitioner(
            compiler_specs=compiler_spec,
            skip_node_id_set=None,
            skip_node_op_set=None,
            skip_mutable_buffer=None,
        )

        for pte_name, exported_program in exported_programs.items():
            logging.debug(f"\nExported program for {pte_name}.pte: {exported_program}")
            exported_program = QnnPassManager().transform_for_export_pipeline(
                exported_program
            )
            transform_passes = QnnPassManager().get_to_edge_transform_passes(
                exported_program
            )
            et_progs[pte_name] = to_edge_transform_and_lower(
                programs=exported_program,
                transform_passes=transform_passes,
                partitioner=[qnn_partitioner],
                constant_methods=None,
                compile_config=qnn_edge_config(),
            ).to_executorch(
                config=ExecutorchBackendConfig(**backend_config_dict),
            )
            logging.debug(
                f"\nExecuTorch program for {pte_name}.pte: {et_progs[pte_name].exported_program().graph_module}"
            )
            delegation_info = get_delegation_info(
                et_progs[pte_name].exported_program().graph_module
            )
            logging.debug(
                f"\nDelegation info Summary for {pte_name}.pte: {delegation_info.get_summary()}"
            )
            logging.debug(
                f"\nDelegation info for {pte_name}.pte: {tabulate(delegation_info.get_operator_delegation_dataframe(), headers='keys', tablefmt='fancy_grid')}"
            )
        return et_progs

    exported_progs = model.export()
    print("model.metadata: ", model.metadata)
    print("len(exported_progs): ", len(exported_progs))
    for pte_name, exported_program in exported_progs.items():
        print(
            "\nExported program for ",
            pte_name,
            ".pte: ",
            len(exported_program.graph_module.graph.nodes),
        )

    et_progs = {}
    backend_config_dict = {}
    compiler_spec = generate_qnn_executorch_compiler_spec(
        soc_model=get_soc_to_chipset_map()[soc],
        backend_options=generate_htp_compiler_spec(use_fp16=True),
    )
    aten_programs = {}
    transform_passes = {}
    qnn_partitioner = QnnPartitioner(
        compiler_specs=compiler_spec,
        skip_node_id_set=None,
        skip_node_op_set=None,
        skip_mutable_buffer=None,
    )

    for pte_name, exported_program in exported_progs.items():
        print(f"\nExported program for {pte_name}.pte")
        print("start QnnPassManager().transform_for_export_pipeline...")
        exported_program = QnnPassManager().transform_for_export_pipeline(
            exported_program
        )
        print("end QnnPassManager().transform_for_export_pipeline...")
        print("start QnnPassManager().get_to_edge_transform_passes...")
        transform_passes = QnnPassManager().get_to_edge_transform_passes(
            exported_program
        )
        print("end QnnPassManager().get_to_edge_transform_passes...")
        print("start to_edge_transform_and_lower...")
        print("to_edge_transform_and_lower: ", to_edge_transform_and_lower)
        et_progs[pte_name] = to_edge_transform_and_lower(
            programs=exported_program,
            transform_passes=transform_passes,
            partitioner=[qnn_partitioner],
            constant_methods=None,
            compile_config=qnn_edge_config(),
        ).to_executorch(
            config=ExecutorchBackendConfig(**backend_config_dict),
        )
        print(
            f"\nExecuTorch program for {pte_name}.pte: {et_progs[pte_name].exported_program().graph_module}"
        )
        delegation_info = get_delegation_info(
            et_progs[pte_name].exported_program().graph_module
        )
        print(
            f"\nDelegation info Summary for {pte_name}.pte: {delegation_info.get_summary()}"
        )
        print(
            f"\nDelegation info for {pte_name}.pte: {tabulate(delegation_info.get_operator_delegation_dataframe(), headers='keys', tablefmt='fancy_grid')}"
        )
    return et_progs

    # return _lower_to_executorch(exported_progs, model.metadata, **kwargs)


def _get_recipe_kwargs(dtype: str, soc: str) -> Dict[str, Any]:
    recipe_kwargs = {
        "dtype": dtype,
        "soc": soc,
    }
    return recipe_kwargs


def _make_recipe(recipe_name, recipe_kwargs):
    @register_recipe(recipe_name)
    def recipe_fn(exported_programs: Dict[str, ExportedProgram], **kwargs):
        print(
            "register_recipe, recipe_name, recipe_kwargs: ", recipe_name, recipe_kwargs
        )
        return _export_to_executorch(
            exported_programs,
            **recipe_kwargs,
        )

    return recipe_fn


# Register recipes for qnn backend
for dtype, soc in product(["fp16"], ["SM8650", "SM8550", "SM8450"]):
    recipe_name = f"qnn_{dtype}"
    recipe_name += f"_{soc}"
    recipe_kwargs = _get_recipe_kwargs(dtype=dtype, soc=soc)
    _make_recipe(recipe_name, recipe_kwargs)
