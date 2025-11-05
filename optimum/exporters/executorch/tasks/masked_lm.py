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

from transformers import AutoModelForMaskedLM

from ..integrations import MaskedLMExportableModule
from ..quantization import quantize_model_
from ..task_registry import register_task


# NOTE: It’s important to map the registered task name to the pipeline name in https://github.com/huggingface/transformers/blob/main/utils/update_metadata.py.
# This will streamline using inferred task names and make exporting models to Hugging Face pipelines easier.
@register_task("fill-mask")
def load_masked_lm_model(model_name_or_path: str, **kwargs) -> MaskedLMExportableModule:
    """
    Loads a seq2seq language model for conditional text generation and registers it under the task
    'fill-mask' using Hugging Face's `AutoModelForMaskedLM`.

    Args:
        model_name_or_path (str):
            Model ID on huggingface.co or path on disk to the model repository to export. For example:
            `model_name_or_path="google-bert/bert-base-uncased"` or `mode_name_or_path="/path/to/model_folder`
        **kwargs:
            Additional configuration options for the model.

    Returns:
        MaskedLMExportableModule:
            An instance of `MaskedLMExportableModule` for exporting and lowering to ExecuTorch.
    """

    eager_model = AutoModelForMaskedLM.from_pretrained(model_name_or_path).to("cpu").eval()

    qlinear_config = kwargs.get("qlinear", None)
    qlinear_packing_format = kwargs.get("qlinear_packing_format", None)
    qembedding_config = kwargs.get("qembedding", None)
    quantize_model_(
        eager_model,
        qlinear_config=qlinear_config,
        qlinear_packing_format=qlinear_packing_format,
        qembedding_config=qembedding_config,
    )

    return MaskedLMExportableModule(eager_model)
