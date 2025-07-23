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

from ..integrations import (
    CausalLMExportableModule,
    MaskedLMExportableModule,
    Seq2SeqLMExportableModule,
)
from ..recipe_registry import register_recipe


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
    import coremltools as ct

    from executorch.backends.apple.coreml.compiler import CoreMLBackend
    from executorch.backends.apple.coreml.partition import CoreMLPartitioner

    def _lower_to_executorch(
        exported_programs: Dict[str, ExportedProgram],
        metadata,
        compute_unit,
        minimum_deployment_target,
        compute_precision,
    ) -> Dict[str, ExecutorchProgram]:
        et_progs = {}
        backend_config_dict = {}
        for pte_name, exported_program in exported_programs.items():
            logging.debug(f"\nExported program for {pte_name}.pte: {exported_program}")
            et_progs[pte_name] = to_edge_transform_and_lower(
                exported_program,
                partitioner=[
                    CoreMLPartitioner(
                        compile_specs=CoreMLBackend.generate_compile_specs(
                            compute_unit=compute_unit,
                            minimum_deployment_target=minimum_deployment_target,
                            compute_precision=compute_precision,
                            model_type=CoreMLBackend.MODEL_TYPE.MODEL,
                        ),
                        take_over_mutable_buffer=(minimum_deployment_target >= ct.target.iOS18),
                    )
                ],
                compile_config=EdgeCompileConfig(
                    _check_ir_validity=False,
                    # In ET 0.7, we can set _skip_dim_order=False
                    _skip_dim_order=True,
                ),
                constant_methods=metadata,
            ).to_executorch(
                config=ExecutorchBackendConfig(**backend_config_dict),
            )
            logging.debug(
                f"\nExecuTorch program for {pte_name}.pte: {et_progs[pte_name].exported_program().graph_module}"
            )
            delegation_info = get_delegation_info(et_progs[pte_name].exported_program().graph_module)
            logging.debug(f"\nDelegation info Summary for {pte_name}.pte: {delegation_info.get_summary()}")
            logging.debug(
                f"\nDelegation info for {pte_name}.pte: {tabulate(delegation_info.get_operator_delegation_dataframe(), headers='keys', tablefmt='fancy_grid')}"
            )
        return et_progs

    exported_progs = model.export()
    return _lower_to_executorch(exported_progs, model.metadata, **kwargs)


def _get_recipe_kwargs(dtype: str, compute_unit: str) -> Dict[str, Any]:
    import coremltools as ct

    compute_precision = {
        "fp16": ct.precision.FLOAT16,
        "fp32": ct.precision.FLOAT32,
    }[dtype]

    compute_unit = {
        "cpu": ct.ComputeUnit.CPU_ONLY,
        "gpu": ct.ComputeUnit.CPU_AND_GPU,
        "ne": ct.ComputeUnit.CPU_AND_NE,
        "all": ct.ComputeUnit.ALL,
    }[compute_unit]

    recipe_kwargs = {
        "compute_precision": compute_precision,
        "compute_unit": compute_unit,
        "minimum_deployment_target": ct.target.iOS18,
    }
    return recipe_kwargs


def _make_recipe(recipe_name, recipe_kwargs):
    @register_recipe(recipe_name)
    def recipe_fn(exported_programs: Dict[str, ExportedProgram], **kwargs):
        return _export_to_executorch(
            exported_programs,
            **recipe_kwargs,
        )

    return recipe_fn


# Register recipes for CoreML backend
for dtype, compute_unit in product(["fp32", "fp16"], ["cpu", "gpu", "ne", "all"]):
    recipe_name = f"coreml_{dtype}"
    if compute_unit != "all":
        recipe_name += f"_{compute_unit}"
    recipe_kwargs = _get_recipe_kwargs(dtype=dtype, compute_unit=compute_unit)
    _make_recipe(recipe_name, recipe_kwargs)
