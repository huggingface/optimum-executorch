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

import torchao
from transformers import AutoConfig, AutoModelForCausalLM, GenerationConfig

from ..integrations import CausalLMExportableModule
from ..quantization import quantize_model_
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
    disable_dynamic_shapes = kwargs.get("disable_dynamic_shapes", False)
    use_custom_sdpa = kwargs.get("use_custom_sdpa", False)
    use_custom_kv_cache = kwargs.get("use_custom_kv_cache", False)
    attn_implementation = kwargs.get("attn_implementation", "custom_sdpa" if use_custom_sdpa else "sdpa")
    cache_implementation = kwargs.get("cache_implementation", "static")
    use_custom_sdpa = use_custom_sdpa or attn_implementation == "custom_sdpa"
    max_seq_len = kwargs.get("max_seq_len", None)
    max_length = max_seq_len if max_seq_len is not None else kwargs.get("max_length", 2048)
    config = kwargs.get("config") or AutoConfig.from_pretrained(model_name_or_path)

    if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
        # NOTE: To make the model exportable we need to set the rope scaling to default to avoid hitting
        # the data-dependent control flow in _longrope_frequency_update. Alternatively, users should rewrite
        # that function to avoid the data-dependent control flow.
        config.rope_scaling["type"] = "default"

    if hasattr(config, "use_cache") and config.use_cache is False:
        config.use_cache = True

    def _load_eager_pretrained(
        model_name_or_path,
        device,
        dtype,
        config,
        attn_implementation,
        cache_implementation,
        batch_size,
        max_length,
    ):
        eager_model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            device_map=device,
            dtype=dtype,
            config=config,
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
        return eager_model

    try:
        eager_model = _load_eager_pretrained(
            model_name_or_path,
            device,
            dtype,
            config,
            attn_implementation,
            cache_implementation,
            batch_size,
            max_length,
        )
    except ValueError as e:
        if "torch.nn.functional.scaled_dot_product_attention" in str(e):
            logging.info("⚠ SDPA attention not supported, falling back to eager implementation")
            attn_implementation = "eager"
            eager_model = _load_eager_pretrained(
                model_name_or_path,
                device,
                dtype,
                config,
                attn_implementation,
                cache_implementation,
                batch_size,
                max_length,
            )
        else:
            raise e

    # Must disable gradient when exporting a model with a prequantized checkpoint,
    # e.g. "pytorch/Phi-4-mini-instruct-8da4w".
    for param in eager_model.parameters():
        if isinstance(param, torchao.utils.TorchAOBaseTensor):
            param.requires_grad = False

    qlinear_config = kwargs.get("qlinear", None)
    qlinear_packing_format = kwargs.get("qlinear_packing_format", None)
    qembedding_config = kwargs.get("qembedding", None)
    quantize_model_(
        eager_model,
        qlinear_config=qlinear_config,
        qlinear_packing_format=qlinear_packing_format,
        qembedding_config=qembedding_config,
    )

    return CausalLMExportableModule(
        eager_model, max_length, use_custom_kv_cache, use_custom_sdpa, disable_dynamic_shapes
    )
