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
from transformers import GenerationConfig, PretrainedConfig, PreTrainedModel, TorchExportableModuleWithStaticCache

from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import (
    EdgeCompileConfig,
    ExecutorchBackendConfig,
    to_edge_transform_and_lower,
)

from ..recipe_registry import register_recipe


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

    def _get_constant_methods(config: PretrainedConfig, generation_config: GenerationConfig, **kwargs):
        metadata = {
            "get_dtype": 5 if config.torch_dtype == torch.float16 else 6,
            "get_bos_id": config.bos_token_id,
            "get_eos_id": config.eos_token_id,
            "get_head_dim": config.hidden_size / config.num_attention_heads,
            "get_n_kv_heads": getattr(config, "num_key_value_heads", None),
            "get_n_layers": config.num_hidden_layers,
            "get_vocab_size": config.vocab_size,
            "get_max_batch_size": generation_config.cache_config.batch_size,
            "get_max_seq_len": generation_config.cache_config.max_cache_len,
            "use_kv_cache": generation_config.use_cache,
        }
        return {k: v for k, v in {**metadata, **kwargs}.items() if v is not None}

    def _lower_to_executorch(exported_program, metadata=None):
        with torch.nn.attention.sdpa_kernel([SDPBackend.MATH]), torch.no_grad():
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
        metadata = _get_constant_methods(model.model.config, model.model.generation_config)

        with torch.nn.attention.sdpa_kernel([SDPBackend.MATH]), torch.no_grad():
            exported_program = torch.export._trace._export(
                model,
                args=(example_input_ids,),
                kwargs={"cache_position": example_cache_position},
                pre_dispatch=False,
                strict=True,
            )
            return {"model": _lower_to_executorch(exported_program, metadata)}
    elif task == "seq2seq-lm":
        exported_model = model.export()
        kwargs = {"max_hidden_seq_length": model.max_hidden_seq_length}
        metadata = _get_constant_methods(model.config, model.generation_config, **kwargs)
        return {
            "encoder": _lower_to_executorch(exported_model.exported_encoder, metadata),
            "decoder": _lower_to_executorch(exported_model.exported_decoder, metadata),
        }

    else:
        # TODO: Prepare model inputs for other tasks
        raise ValueError(f"Unsupported task '{task}'.")
