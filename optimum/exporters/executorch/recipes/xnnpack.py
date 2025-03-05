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

from typing import Union

import torch
import torch.export._trace
from torch.nn.attention import SDPBackend
from transformers import PreTrainedModel, TorchExportableModuleWithStaticCache

from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import (
    EdgeCompileConfig,
    ExecutorchBackendConfig,
    to_edge_transform_and_lower,
)

from ..recipe_registry import register_recipe
from ..utils import save_config_to_constant_methods


@register_recipe("xnnpack")
def export_to_executorch_with_xnnpack(
    model: Union[PreTrainedModel, TorchExportableModuleWithStaticCache],
    task: str,
    **kwargs,
):
    """
    Export a PyTorch model to ExecuTorch w/ delegation to XNNPACK backend.

    This function also write metadata required by the ExecuTorch runtime to the model.

    Args:
        model (Union[PreTrainedModel, TorchExportableModuleWithStaticCache]):
            The PyTorch model to be exported to ExecuTorch.
        task (str):
            The task name to export the model for (e.g., "text-generation").
        **kwargs:
            Additional keyword arguments for recipe-specific configurations.

    Returns:
        ExecuTorchProgram:
            The exported and optimized program for ExecuTorch.
    """

    def _lower_to_executorch(exported_program, metadata=None):
        return to_edge_transform_and_lower(
            exported_program,
            partitioner=[XnnpackPartitioner()],
            compile_config=EdgeCompileConfig(_skip_dim_order=True),
            constant_methods=metadata,
        ).to_executorch(
            config=ExecutorchBackendConfig(
                extract_delegate_segments=True,
            ),
        )

    metadata = {}
    if task == "text-generation":
        example_input_ids = torch.tensor([[1]], dtype=torch.long)
        example_cache_position = torch.tensor([0], dtype=torch.long)
        metadata = save_config_to_constant_methods(model.model.config, model.model.generation_config)

        with torch.nn.attention.sdpa_kernel([SDPBackend.MATH]), torch.no_grad():
            exported_program = torch.export._trace._export(
                model,
                args=(example_input_ids,),
                kwargs={"cache_position": example_cache_position},
                pre_dispatch=False,
                strict=True,
            )
            et_model = _lower_to_executorch(exported_program, metadata)
            return {"model": et_model}
    elif task == "fill-mask":
        max_position_embeddings = getattr(model.config, "max_position_embeddings", 64)
        max_seq_length = max(max_position_embeddings - 1, 1)
        # Create dummy inputs with expected shapes
        batch_size = 1
        seq_length = max_seq_length
        vocab_size = model.config.vocab_size

        # Create example inputs (no need for tokenizer)
        dummy_input_ids = torch.randint(0, vocab_size, (batch_size, seq_length), dtype=torch.long)
        dummy_attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)

        # Define dynamic dimensions using Dim class
        seq_dim = torch.export.Dim("sequence_length", min=1, max=max_seq_length)  # Allow sequences up to max_length

        # Define dynamic shapes with Dim objects
        dynamic_shapes = {
            "input_ids": {1: seq_dim},
            "attention_mask": {1: seq_dim},
        }

        # Export the model with dynamic dimensions
        with torch.no_grad():
            metadata = save_config_to_constant_methods(model.config, model.generation_config)
            exported_program = torch.export.export(
                model,
                args=(dummy_input_ids,),
                kwargs={"attention_mask": dummy_attention_mask},
                dynamic_shapes=dynamic_shapes,
            )
            et_prog = to_edge_transform_and_lower(
                exported_program,
                partitioner=[XnnpackPartitioner()],
                compile_config=EdgeCompileConfig(_skip_dim_order=True),
                constant_methods=metadata,
            ).to_executorch(
                config=ExecutorchBackendConfig(
                    extract_delegate_segments=True,
                ),
            )
            return {"model": et_prog}
    elif task == "text2text-generation":
        exported_model = model.export()
        kwargs = {"max_hidden_seq_length": model.max_hidden_seq_length}
        metadata = save_config_to_constant_methods(model.config, model.generation_config, **kwargs)
        with torch.nn.attention.sdpa_kernel([SDPBackend.MATH]), torch.no_grad():
            encoder_model = _lower_to_executorch(exported_model.exported_encoder, metadata)
            decoder_model = _lower_to_executorch(exported_model.exported_decoder, metadata)
            return {
                "encoder": encoder_model,
                "decoder": decoder_model,
            }

    else:
        # TODO: Prepare model inputs for other tasks
        raise ValueError(f"Unsupported task '{task}'.")
