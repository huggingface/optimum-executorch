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
from typing import Optional

import torch


def quantize_model_(
    eager_model: torch.nn.Module,
    qlinear_config: Optional[str] = None,
    qlinear_group_size: Optional[int] = 32,
    qembedding_config: Optional[str] = None,
    qembedding_group_size: Optional[int] = 0,
) -> torch.nn.Module:
    if not (qlinear_config or qembedding_config):
        return

    from torchao.quantization.granularity import PerAxis, PerGroup
    from torchao.quantization.quant_api import (
        Int8DynamicActivationIntxWeightConfig,
        IntxWeightOnlyConfig,
        quantize_,
    )
    from torchao.utils import unwrap_tensor_subclass

    if qembedding_group_size == 0:
        embedding_weight_granularity = PerAxis(0)
    else:
        embedding_weight_granularity = PerGroup(qembedding_group_size)
    if qembedding_config:
        logging.info("Quantizing embedding layers.")
        embedding_config = {
            "4w": IntxWeightOnlyConfig(
                weight_dtype=torch.int4,
                granularity=embedding_weight_granularity,
            ),
            "8w": IntxWeightOnlyConfig(
                weight_dtype=torch.int8,
                granularity=embedding_weight_granularity,
            ),
        }[qembedding_config]

        # TODO: Should switch to `AOPerModuleConfig` once fix for tied weights is available.
        quantize_(
            eager_model,
            embedding_config,
            lambda m, fqn: isinstance(m, torch.nn.Embedding),
        )

    if qlinear_group_size == 0:
        linear_weight_granularity = PerAxis(0)
    else:
        linear_weight_granularity = PerGroup(qlinear_group_size)
    if qlinear_config:
        logging.info("Quantizing linear layers.")
        linear_config = {
            "8da4w": Int8DynamicActivationIntxWeightConfig(
                weight_dtype=torch.int4,
                weight_granularity=linear_weight_granularity,
            ),
            "4w": IntxWeightOnlyConfig(
                weight_dtype=torch.int4,
                granularity=linear_weight_granularity,
            ),
            "8w": IntxWeightOnlyConfig(
                weight_dtype=torch.int8,
                granularity=linear_weight_granularity,
            ),
        }[qlinear_config]
        quantize_(
            eager_model,
            linear_config,
        )

    unwrap_tensor_subclass(eager_model)
