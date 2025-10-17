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

from transformers import AutoModelForSeq2SeqLM

from ..integrations import Seq2SeqLMExportableModule
from ..task_registry import register_task


# NOTE: It’s important to map the registered task name to the pipeline name in https://github.com/huggingface/transformers/blob/main/utils/update_metadata.py.
# This will streamline using inferred task names and make exporting models to Hugging Face pipelines easier.
@register_task("text2text-generation")
def load_seq2seq_lm_model(model_name_or_path: str, **kwargs) -> Seq2SeqLMExportableModule:
    """
        Loads a seq2seq language model for conditional text generation and registers it under the task
        'text2text-generation' using Hugging Face's `AutoModelForSeq2SeqLM`.

        Args:
            model_name_or_path (str):
                Model ID on huggingface.co or path on disk to the model repository to export. For example:
                `model_name_or_path="google-t5/t5-small"` or `mode_name_or_path="/path/to/model_folder`
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
    n"""
    device = "cpu"
    batch_size = 1
    max_hidden_seq_len = kwargs.get("max_hidden_seq_len", 4096)
    max_seq_len = kwargs.get("max_seq_len", 1024)

    full_model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path).to(device).eval()
    return Seq2SeqLMExportableModule(
        full_model,
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        max_hidden_seq_len=max_hidden_seq_len,
    )
