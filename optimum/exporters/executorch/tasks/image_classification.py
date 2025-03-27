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

from transformers import AutoModelForImageClassification

from ..integrations import VisionEncoderExportableModule
from ..task_registry import register_task


# NOTE: It’s important to map the registered task name to the pipeline name in https://github.com/huggingface/transformers/blob/main/utils/update_metadata.py.
# This will streamline using inferred task names and make exporting models to Hugging Face pipelines easier.
@register_task("image-classification")
def load_image_classification_model(model_name_or_path: str, **kwargs) -> VisionEncoderExportableModule:
    """
    Loads a vision model for image classification and registers it under the task
    'image-classification' using Hugging Face's `AutoModelForImageClassification`.

    Args:
        model_name_or_path (str):
            Model ID on huggingface.co or path on disk to the model repository to export. For example:
            `model_name_or_path="google/vit-base-patch16-224"` or `mode_name_or_path="/path/to/model_folder`
        **kwargs:
            Additional configuration options for the model.

    Returns:
        VisionEncoderExportableModule:
            An instance of `VisionEncoderExportableModule` for exporting and lowering to ExecuTorch.
    """

    eager_model = AutoModelForImageClassification.from_pretrained(model_name_or_path, **kwargs).to("cpu").eval()
    return VisionEncoderExportableModule(eager_model)
