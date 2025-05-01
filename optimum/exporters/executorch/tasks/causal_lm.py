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

import torch
import torchao
from packaging.version import parse
from transformers import AutoModelForCausalLM, GenerationConfig

from ..integrations import CausalLMExportableModule
from ..task_registry import register_task


# NOTE: It’s important to map the registered task name to the pipeline name in https://github.com/huggingface/transformers/blob/main/utils/update_metadata.py.
# This will streamline using inferred task names and make exporting models to Hugging Face pipelines easier.
@register_task("text-generation")
def load_causal_lm_model(model_name_or_path: str, **kwargs) -> CausalLMExportableModule:
    """
    Loads a causal language model for text generation and registers it under the task
    'text-generation' using Hugging Face's AutoModelForCausalLM.

    Args:
        model_name_or_path (str):
            Model ID on huggingface.co or path on disk to the model repository to export. For example:
            `model_name_or_path="meta-llama/Llama-3.2-1B"` or `mode_name_or_path="/path/to/model_folder`
        **kwargs:
            Additional configuration options for the model:
                - dtype (str, optional):
                    Data type for model weights (default: "float32").
                    Options include "float16" and "bfloat16".
                - attn_implementation (str, optional):
                    Attention mechanism implementation (default: "sdpa").
                - cache_implementation (str, optional):
                    Cache management strategy (default: "static").
                - max_length (int, optional):
                    Maximum sequence length for generation (default: 2048).

    Returns:
        CausalLMExportableModule:
            An instance of `CausalLMExportableModule` for exporting and lowering to ExecuTorch.
    """
    device = "cpu"
    batch_size = 1
    dtype = kwargs.get("dtype", "float32")
    use_custom_sdpa = kwargs.get("use_custom_sdpa", False)
    attn_implementation = kwargs.get("attn_implementation", "custom_sdpa" if use_custom_sdpa else "sdpa")
    cache_implementation = kwargs.get("cache_implementation", "static")
    max_length = kwargs.get("max_length", 2048)
    config = kwargs.get("config", None)
    quantization_recipe = kwargs.get("quantize", None)

    eager_model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        device_map=device,
        torch_dtype=dtype,
        config=config,
        # quantization_config=quantization_config,
        attn_implementation=attn_implementation,
        generation_config=GenerationConfig(
            use_cache=True,
            cache_implementation=cache_implementation,
            max_length=max_length,
            cache_config={
                "batch_size": batch_size,
                "max_cache_len": max_length,
            },
        ),
    )

    if quantization_recipe == "8da4w":
        if parse(torchao.__version__) < parse("0.11.0.dev0"):
            raise RuntimeError("Quantization 8da4w requires torchao >= 0.11.0. Please upgrade torchao.")

        from torchao.quantization.granularity import PerGroup
        from torchao.quantization.quant_api import (
            Int8DynamicActivationIntxWeightConfig,
        )

        # TODO: Should switch to TorchAoConfig once the quant issue on final lm_head layer is fixed.
        linear_config = Int8DynamicActivationIntxWeightConfig(
            weight_dtype=torch.int4,
            weight_granularity=PerGroup(64),
        )

        torchao.quantize_(
            eager_model,
            linear_config,
        )

    return CausalLMExportableModule(eager_model)
