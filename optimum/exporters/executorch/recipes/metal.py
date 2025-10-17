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

from executorch import version as executorch_version


EXECUTORCH_VERSION = parse(executorch_version.__version__)
METAL_BACKEND_AVAILABLE = EXECUTORCH_VERSION >= parse("1.1.0.dev20251017")

if METAL_BACKEND_AVAILABLE:
    try:
        from executorch.backends.apple.metal.metal_backend import MetalBackend
        from executorch.backends.apple.metal.metal_partitioner import MetalPartitioner
    except ImportError:
        METAL_BACKEND_AVAILABLE = False

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

        exported_progs = model.export()

        return _lower_to_executorch(exported_progs, model.metadata)
