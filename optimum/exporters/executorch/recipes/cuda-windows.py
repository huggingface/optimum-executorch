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

from typing import Union

from ..integrations import (
    CausalLMExportableModule,
    MaskedLMExportableModule,
    MultiModalTextToTextExportableModule,
    Seq2SeqLMExportableModule,
)
from ..recipe_registry import register_recipe
from .cuda import lower_to_executorch


@register_recipe("cuda-windows")
def export_to_executorch_with_cuda_windows(
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

    return lower_to_executorch(
        exported_progs, model.metadata, is_windows=True, model_config=getattr(model, "config", None)
    )
