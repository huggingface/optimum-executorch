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

    if qembedding_config:
        if qlinear_config == "8w":
            assert (
                qembedding_group_size == 0
            ), "8-bit embedding quantization only supports per-token at the moment, please use qembedding_group_size = 0."
        if qembedding_group_size == 0:
            embedding_weight_granularity = PerAxis(0)
        else:
            assert qembedding_group_size % 2 == 0, "Embedding quantization group size must be a multiple of 2."
            embedding_weight_granularity = PerGroup(qembedding_group_size)

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

    if qlinear_config:

        def build_linear_config(config_key: str, granularity):
            if config_key == "8da4w":
                return Int8DynamicActivationIntxWeightConfig(
                    weight_dtype=torch.int4,
                    weight_granularity=granularity,
                )
            if config_key == "4w":
                return IntxWeightOnlyConfig(
                    weight_dtype=torch.int4,
                    granularity=granularity,
                )
            if config_key == "8w":
                return IntxWeightOnlyConfig(
                    weight_dtype=torch.int8,
                    granularity=granularity,
                )
            if config_key == "8da8w":
                return Int8DynamicActivationIntxWeightConfig(
                    weight_dtype=torch.int8,
                    weight_granularity=PerAxis(0),
                )
            raise ValueError(f"Unsupported linear quantization config '{config_key}'.")

        qlinear_configs = [cfg.strip() for cfg in qlinear_config.split(",")]
        if any(cfg == "" for cfg in qlinear_configs):
            raise ValueError("Linear quantization config entries must be non-empty.")
        if len(qlinear_configs) > 2:
            raise ValueError("Expected at most one fallback linear quantization config, got more than one comma.")

        primary_linear_config_key = qlinear_configs[0]
        fallback_linear_config_key = qlinear_configs[1] if len(qlinear_configs) == 2 else None

        if qlinear_group_size == 0:
            linear_weight_granularity = PerAxis(0)
            if fallback_linear_config_key is not None:
                logging.warning(
                    "qlinear_group_size is 0, fallback linear config will not be used as all layers will be quantized with per-axis granularity."
                )
                fallback_linear_config_key = None
        else:
            assert (
                qlinear_group_size % 2 == 0
            ), f"Linear quantization group size must be a multiple of 2, got {qlinear_group_size}."
            linear_weight_granularity = PerGroup(qlinear_group_size)

        logging.info("Quantizing linear layers.")
        primary_linear_config = build_linear_config(primary_linear_config_key, linear_weight_granularity)

        # First, quantize layers that are compatible with group quantization
        def per_group_filter(module, fqn):
            if isinstance(module, torch.nn.Linear):
                # Check if hidden dimension is divisible by group size
                # For Linear layers, weight shape is [out_features, in_features]
                # Group quantization typically applies to the in_features dimension (dim=1)
                return qlinear_group_size == 0 or (module.weight.shape[1] % qlinear_group_size == 0)
            return False

        quantize_(
            eager_model,
            primary_linear_config,
            filter_fn=per_group_filter,
        )

        # Then, quantize incompatible layers using the fallback per-axis config
        if fallback_linear_config_key is not None:
            fallback_linear_config = build_linear_config(fallback_linear_config_key, PerAxis(0))

            def per_token_filter(module, fqn):
                if isinstance(module, torch.nn.Linear):
                    return module.weight.shape[1] % qlinear_group_size != 0
                return False

            logging.info(
                f"Applying fallback linear config '{fallback_linear_config_key}' (per-axis)"
                f" to layers incompatible with group size {qlinear_group_size}."
            )
            quantize_(
                eager_model,
                fallback_linear_config,
                filter_fn=per_token_filter,
            )

    unwrap_tensor_subclass(eager_model)
