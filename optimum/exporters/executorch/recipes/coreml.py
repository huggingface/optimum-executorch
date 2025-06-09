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
from typing import Dict, Union

from packaging.version import parse
from tabulate import tabulate
from torch.export import ExportedProgram
import coremltools as ct
import torch

from executorch import version as executorch_version
from executorch.backends.apple.coreml.partition import CoreMLPartitioner
from executorch.backends.apple.coreml.compiler import CoreMLBackend

from executorch.devtools.backend_debug import get_delegation_info
from executorch.exir import (
    EdgeCompileConfig,
    ExecutorchBackendConfig,
    ExecutorchProgram,
    to_edge_transform_and_lower,
)
from optimum.executorch.passes.remove_padding_idx_embedding_pass import RemovePaddingIdxEmbeddingPass
from executorch.backends.apple.coreml.quantizer import CoreMLQuantizer
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e
from ..integrations import (
    CausalLMExportableModule,
    MaskedLMExportableModule,
    Seq2SeqLMExportableModule,
)
from ..recipe_registry import register_recipe

def get_quantization_config():
    quantization_config = ct.optimize.torch.quantization.LinearQuantizerConfig.from_dict(
        {
            "global_config": {
                "quantization_scheme": ct.optimize.torch.quantization.QuantizationScheme.symmetric,
                "activation_dtype": torch.quint8,
                "weight_dtype": torch.qint8,
                "weight_per_channel": True,
            }
        }
    )
    return quantization_config

def quantize_program(ep):
    quantizer = CoreMLQuantizer(get_quantization_config())
    gm = ep.module()
    
    args, kwargs = ep.example_inputs
    prepared_model = prepare_pt2e(gm, quantizer) 
    prepared_model(*args, **kwargs)
    converted_model = convert_pt2e(prepared_model)
    return torch.export.export(converted_model, args, kwargs)


@register_recipe("coreml")
def export_to_executorch_with_coreml(
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

    def _lower_to_executorch(
        exported_programs: Dict[str, ExportedProgram],
        metadata=None,
        **kwargs,
    ) -> Dict[str, ExecutorchProgram]:

        minimum_deployment_target = kwargs.get("minimum_ios_deployment_target", "15")
        minimum_deployment_target = {
            "15": ct.target.iOS15,
            "16": ct.target.iOS16,
            "17": ct.target.iOS17,
            "18": ct.target.iOS18,
        }[minimum_deployment_target]

        compute_precision = kwargs.get("compute_precision", "fp16")
        compute_precision = {
            "fp16": ct.precision.FLOAT16,
            "fp32": ct.precision.FLOAT32,
        }[compute_precision]

        model_type = kwargs.get("model_type", "model")
        model_type = {
            "model": CoreMLBackend.MODEL_TYPE.MODEL,
            "modelc": CoreMLBackend.MODEL_TYPE.COMPILED_MODEL,
        }[model_type]
        take_over_mutable_buffer = kwargs.get("take_over_mutable_buffer", True)
        quantize = kwargs.get("quantize", False)

        et_progs = {}
        backend_config_dict = {}
        for pte_name, exported_program in exported_programs.items():
            logging.debug(f"\nExported program for {pte_name}.pte: {exported_program}")
            if quantize:
                exported_program = quantize_program(exported_program)
            et_progs[pte_name] = to_edge_transform_and_lower(
                exported_program,
                partitioner=[CoreMLPartitioner(
                    compile_specs=CoreMLBackend.generate_compile_specs(
                        minimum_deployment_target=minimum_deployment_target,
                        compute_precision=compute_precision,
                        model_type=model_type,
                    ),
                    take_over_mutable_buffer=take_over_mutable_buffer, # Fails when set to true
                )],
                compile_config=EdgeCompileConfig(
                    _check_ir_validity=False,
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
