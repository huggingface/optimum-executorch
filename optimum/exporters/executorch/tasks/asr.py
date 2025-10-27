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
import torchao
from transformers import AutoModelForSpeechSeq2Seq

from ..integrations import Seq2SeqLMExportableModule
from ..quantization import quantize_model_
from ..task_registry import register_task


# NOTE: It’s important to map the registered task name to the pipeline name in https://github.com/huggingface/transformers/blob/main/utils/update_metadata.py.
# This will streamline using inferred task names and make exporting models to Hugging Face pipelines easier.
@register_task("automatic-speech-recognition")
def load_seq2seq_speech_model(model_name_or_path: str, **kwargs) -> Seq2SeqLMExportableModule:
    """
    Loads a model for speech seq2seq and registers it under the task
    'automatic-speech-recognition' using Hugging Face's `AutoModelForSpeechSeq2Seq`.

    Args:
        model_name_or_path (str):
            Model ID on huggingface.co or path on disk to the model repository to export. For example:
            `model_name_or_path="openai/whisper-tiny"` or `mode_name_or_path="/path/to/model_folder`
        **kwargs:
            Additional configuration options for the model:
                - dtype (str, optional):
                    Data type for model weights (default: "float32").
                    Options include "float16" and "bfloat16".
                - max_hidden_seq_length (int, optional):
                    Maximum hidden sequence length (default: 4096).
                - max_cache_length (int, optional):
                    Maximum sequence length for generation (default: 1024).

    Returns:
        Seq2SeqLMExportableModule:
            An instance of `Seq2SeqLMExportableModule` for exporting and lowering to ExecuTorch.
    """
    device = kwargs.get("device", "cpu")
    batch_size = 1
    max_hidden_seq_len = kwargs.get("max_hidden_seq_len", 4096)
    max_seq_len = kwargs.get("max_seq_len", 1024)
    dtype = kwargs.get("dtype", "float32")

    full_model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name_or_path, dtype=dtype, device_map=device).eval()

    for param in full_model.parameters():
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

    # Quantize decoder linear weights.
    quantize_decoder_kwargs = {
        "eager_model": getattr(full_model.model, "decoder"),
        "qlinear_config": qlinear_config,
    }
    if qlinear_group_size is not None:
        quantize_decoder_kwargs["qlinear_group_size"] = qlinear_group_size
    if qlinear_packing_format is not None:
        quantize_decoder_kwargs["qlinear_packing_format"] = qlinear_packing_format
    quantize_model_(**quantize_decoder_kwargs)

    # Quantize encoder linear weights.
    quantize_encoder_kwargs = {
        "eager_model": getattr(full_model.model, "encoder"),
        "qlinear_config": qlinear_encoder_config,
    }
    if qlinear_encoder_group_size is not None:
        quantize_encoder_kwargs["qlinear_group_size"] = qlinear_encoder_group_size
    if qlinear_encoder_packing_format is not None:
        quantize_encoder_kwargs["qlinear_packing_format"] = qlinear_encoder_packing_format
    quantize_model_(**quantize_encoder_kwargs)

    # Quantize decoder embeddings.
    quantize_decoder_embedding_kwargs = {
        "eager_model": full_model,
        "qembedding_config": qembedding_config,
    }
    if qembedding_group_size is not None:
        quantize_decoder_embedding_kwargs["qembedding_group_size"] = qembedding_group_size
    quantize_model_(**quantize_decoder_embedding_kwargs)

    return Seq2SeqLMExportableModule(
        full_model,
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        max_hidden_seq_len=max_hidden_seq_len,
    )
