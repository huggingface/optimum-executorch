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


import json
import logging
import os.path

import torchao
from transformers import AutoConfig, AutoModelForPreTraining, GenerationConfig

from ..integrations import MultiModalTextToTextExportableModule
from ..quantization import quantize_model_
from ..task_registry import register_task


# NOTE: It's important to map the registered task name to the pipeline name in https://github.com/huggingface/transformers/blob/main/utils/update_metadata.py.
# This will streamline using inferred task names and make exporting models to Hugging Face pipelines easier.
@register_task("image-text-to-text")
@register_task("multimodal-text-to-text")  # Fake task name, since audio-text-to-text is not available.
def load_multimodal_text_to_text_model(model_name_or_path: str, **kwargs):
    """
    Loads a causal language model for multimodal generation (e.g. image-to-text) generation and registers it under the appropriate task
    (e.g. 'image-text-to-text') using Hugging Face's AutoModelForCausalLM.

    Args:
        model_name_or_path (str):
            Model ID on huggingface.co or path on disk to the model repository to export. For example:
            `model_name_or_path="google/gemma-3-4b-it"` or `model_name_or_path="/path/to/model_folder`
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
        MultiModalTextToTextExportableModule:
            An instance of `MultiModalTextToTextExportableModule` for exporting and lowering to ExecuTorch.
    """
    device = kwargs.get("device", "cpu")
    batch_size = 1
    dtype = kwargs.get("dtype", "float32")
    use_custom_sdpa = kwargs.get("use_custom_sdpa", False)
    use_custom_kv_cache = kwargs.get("use_custom_kv_cache", False)
    attn_implementation = kwargs.get("attn_implementation", "custom_sdpa" if use_custom_sdpa else "sdpa")
    cache_implementation = kwargs.get("cache_implementation", "static")
    use_custom_sdpa = use_custom_sdpa or attn_implementation == "custom_sdpa"
    max_seq_len = kwargs.get("max_seq_len", None)
    max_length = max_seq_len if max_seq_len is not None else kwargs.get("max_length", 2048)
    config = kwargs.get("config") or AutoConfig.from_pretrained(model_name_or_path)

    # Load preprocessor_config.json if it exists
    processor_config = None
    # Check if model_name_or_path is a local directory
    if os.path.isdir(model_name_or_path):
        preprocessor_config_path = os.path.join(model_name_or_path, "preprocessor_config.json")
    else:
        # For Hugging Face model IDs, try to find it in the cached directory
        try:
            from transformers.utils import cached_file

            preprocessor_config_path = cached_file(model_name_or_path, "preprocessor_config.json")
        except Exception:
            preprocessor_config_path = None

    if preprocessor_config_path and os.path.exists(preprocessor_config_path):
        try:
            with open(preprocessor_config_path, "r") as f:
                processor_config = json.load(f)
        except (OSError, json.JSONDecodeError):
            processor_config = None

    # Make sure config has text_config.
    if not (hasattr(config, "text_config")):
        raise ValueError(f"The model {model_name_or_path} does not have a `text_config`.")

    if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
        # NOTE: Avoid hitting the data-dependent control flow in _longrope_frequency_update.
        config.rope_scaling["type"] = "default"
    if hasattr(config, "use_cache") and config.use_cache is False:
        config.use_cache = True

    # Using `AutoModelForPreTraining` since it usually routes to the correct model variant and there is no
    # auto model class that captures both audio and image.
    # The correct model variant we are looking for is <Model>ForConditionalGeneration, since it is the top-level
    # model and thus will always contain the necessary model components. As an example of why this is needed,
    # if you just use Gemma3Model instead of Gemma3ForConditionalGeneration, Gemma3Model (which is the decoder part)
    # will not contain the LM head, which is only applied in the latter.
    eager_model = AutoModelForPreTraining.from_pretrained(
        model_name_or_path,
        device_map=device,
        dtype=dtype,
        config=config,
        attn_implementation=attn_implementation,
    )
    eager_model.generation_config = GenerationConfig(
        use_cache=True,
        cache_implementation=cache_implementation,
        max_length=max_length,
        cache_config={
            "batch_size": batch_size,
            "max_cache_len": max_length,
            "device": device,
        },
    )

    # Find the modality (only one modality of either image/audio is supported at the moment).
    if len(eager_model.input_modalities) != 2:
        raise AttributeError(
            "Only one modality is supported for multimodal models at the moment. The input modalities must be either ['text', 'image'] or ['text, 'audio']"
        )
    for input_modality in eager_model.input_modalities:
        if input_modality == "text":
            continue
        modality = input_modality
    eager_encoder = eager_model.get_encoder(modality)

    # Need to do this since apparently when nested modules (e.g. model.language_model) access the .property
    # config, it always comes from the generation_config.json file, not the `generation_config` override
    # from from_pretrained().
    eager_model.get_decoder().generation_config = eager_model.generation_config

    # Must disable gradient when exporting a model with a prequantized checkpoint,
    # e.g. "pytorch/Phi-4-mini-instruct-8da4w".
    for param in eager_model.parameters():
        if isinstance(param, torchao.utils.TorchAOBaseTensor):
            param.requires_grad = False

    qlinear_config = kwargs.get("qlinear", None)
    qlinear_group_size = kwargs.get("qlinear_group_size", None)
    qlinear_packing_format = kwargs.get("qlinear_packing_format", None)
    qlinear_encoder_config = kwargs.get("qlinear_encoder", None)
    qlinear_encoder_group_size = kwargs.get("qlinear_encoder_group_size", None)
    qlinear_encoder_packing_format = kwargs.get("qlinear_encoder_packing_format", None)
    qembedding_config = kwargs.get("qembedding", None)
    qembedding_group_size = kwargs.get("qembedding_group_size", None)
    qembedding_encoder_config = kwargs.get("qembedding_encoder", None)
    qembedding_encoder_group_size = kwargs.get("qembedding_encoder_group_size", None)

    # Quantize decoder linear weights.
    if qlinear_config:
        logging.info("Quantizing decoder linears...")
    quantize_decoder_kwargs = {
        "eager_model": eager_model.get_decoder(),
        "qlinear_config": qlinear_config,
    }
    if qlinear_group_size is not None:
        quantize_decoder_kwargs["qlinear_group_size"] = qlinear_group_size
    if qlinear_packing_format is not None:
        quantize_decoder_kwargs["qlinear_packing_format"] = qlinear_packing_format
    quantize_model_(**quantize_decoder_kwargs)

    # Quantize lm head, if it is separate from the decoder model.
    # e.g. Sometimes  the top-level model will have:
    # def __init__(self, ...):
    #     self.decoder = ...
    #     self.lm_head = ...  # lm_head is not part of the decoder instance
    #     ...
    if not hasattr(eager_model.get_decoder(), "lm_head"):
        # Voxtral specifically is weird since you need to specifically do eager_model.language_model.lm_head.
        lm_head = getattr(eager_model, "lm_head", None) or getattr(eager_model.language_model, "lm_head", None)
        if not lm_head:
            raise AttributeError(
                f"Could not find `lm_head` for {model_name_or_path} has no `lm_head`, please double check if this is expected."
            )
        quantize_lm_head_kwargs = {
            "eager_model": lm_head,
            "qlinear_config": qlinear_config,
        }
        quantize_model_(**quantize_lm_head_kwargs)

    # Quantize encoder linear weights.
    if qlinear_encoder_config:
        logging.info("Quantizing encoder linears...")
    quantize_encoder_kwargs = {
        "eager_model": eager_encoder,
        "qlinear_config": qlinear_encoder_config,
    }
    if qlinear_encoder_group_size is not None:
        quantize_encoder_kwargs["qlinear_group_size"] = qlinear_encoder_group_size
    if qlinear_encoder_packing_format is not None:
        quantize_encoder_kwargs["qlinear_packing_format"] = qlinear_encoder_packing_format
    quantize_model_(**quantize_encoder_kwargs)

    # Quantize decoder embeddings.
    if qembedding_config:
        logging.info("Quantizing embeddings...")
    quantize_decoder_embedding_kwargs = {
        "eager_model": eager_model.get_decoder(),
        "qembedding_config": qembedding_config,
    }
    if qembedding_group_size is not None:
        quantize_decoder_embedding_kwargs["qembedding_group_size"] = qembedding_group_size
    quantize_model_(**quantize_decoder_embedding_kwargs)

    # Quantize encoder embeddings.
    if qembedding_encoder_config:
        logging.info("Quantizing embeddings...")
    quantize_encoder_embedding_kwargs = {
        "eager_model": eager_encoder,
        "qembedding_config": qembedding_encoder_config,
    }
    if qembedding_encoder_group_size is not None:
        quantize_encoder_embedding_kwargs["qembedding_group_size"] = qembedding_encoder_group_size
    quantize_model_(**quantize_encoder_embedding_kwargs)

    return MultiModalTextToTextExportableModule(
        model=eager_model,
        modality=(
            "vision" if modality == "image" else modality
        ),  # TODO: hack since downstream uses "vision" atm. Change this to match Transformers.
        encoder_model=eager_encoder,
        max_seq_len=max_length,
        processor_config=processor_config,
        use_custom_kv_cache=use_custom_kv_cache,
        use_custom_sdpa=use_custom_sdpa,
    )
