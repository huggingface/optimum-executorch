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

"""Defines the command line for the export with ExecuTorch."""

from pathlib import Path
from typing import TYPE_CHECKING

from ...exporters import TasksManager
from ..base import BaseOptimumCLICommand, CommandInfo


if TYPE_CHECKING:
    from argparse import ArgumentParser


def parse_args_executorch(parser):
    required_group = parser.add_argument_group("Required arguments")
    required_group.add_argument(
        "-m",
        "--model",
        type=str,
        required=True,
        help="Model ID on huggingface.co or path on disk to load model from.",
    )
    required_group.add_argument(
        "-o",
        "--output_dir",
        type=Path,
        help="Path indicating the directory where to store the generated ExecuTorch model.",
    )
    required_group.add_argument(
        "--task",
        type=str,
        default="text-generation",
        help=(
            "The task to export the model for. Available tasks depend on the model, but are among:"
            f" {str(TasksManager.get_all_tasks())}."
        ),
    )
    required_group.add_argument(
        "--recipe",
        type=str,
        default="xnnpack",
        help='Pre-defined recipes for export to ExecuTorch. Defaults to "xnnpack".',
    )
    required_group.add_argument(
        "--use_custom_sdpa",
        required=False,
        action="store_true",
        help="For decoder-only models to use custom sdpa with static kv cache to boost performance. Defaults to False.",
    )
    required_group.add_argument(
        "--use_custom_kv_cache",
        required=False,
        action="store_true",
        help="For decoder-only models to use custom kv cache for static cache that updates cache using custom op. Defaults to False.",
    )
    required_group.add_argument(
        "--disable_dynamic_shapes",
        required=False,
        action="store_true",
        help="When this flag is set on decoder-only models, dynamic shapes are disabled during export.",
    )
    required_group.add_argument(
        "--qlinear",
        type=str,
        choices=["8da4w", "4w", "8w"],
        required=False,
        help=(
            "Quantization config for decoder linear layers.\n\n"
            "Options:\n"
            "  8da4w - 8-bit dynamic activation, 4-bit weight\n"
            "  4w    - 4-bit weight only\n"
            "  8w    - 8-bit weight only"
        ),
    )
    required_group.add_argument(
        "--qlinear_group_size", type=int, required=False, help="Group size for decoder linear quantization."
    )
    required_group.add_argument(
        "--qlinear_packing_format",
        type=str,
        choices=["tile_packed_to_4d"],
        required=False,
        help=(
            "Packing format for decoder linear layers.\n"
            "Only applicable to certain backends such as CUDA and Metal\n\n"
            "Options:\n"
            "  tile_packed_to_4d  - int4 4d packing format"
        ),
    )
    required_group.add_argument(
        "--qlinear_encoder",
        type=str,
        choices=["8da4w", "4w", "8w"],
        required=False,
        help=(
            "Quantization config for encoder linear layers.\n\n"
            "Options:\n"
            "  8da4w - 8-bit dynamic activation, 4-bit weight\n"
            "  4w    - 4-bit weight only\n"
            "  8w    - 8-bit weight only"
        ),
    )
    required_group.add_argument(
        "--qlinear_encoder_group_size", type=int, required=False, help="Group size for encoder linear quantization."
    )
    required_group.add_argument(
        "--qlinear_encoder_packing_format",
        type=str,
        choices=["tile_packed_to_4d"],
        required=False,
        help=(
            "Packing format for encoder linear layers.\n"
            "Only applicable to certain backends such as CUDA and Metal\n\n"
            "Options:\n"
            "  tile_packed_to_4d  - int4 4d packing format"
        ),
    )
    required_group.add_argument(
        "--qembedding",
        type=str,
        choices=["4w", "8w"],
        required=False,
        help=(
            "Quantization config for embedding layer.\n\n"
            "Options:\n"
            "  4w    - 4-bit weight only\n"
            "  8w    - 8-bit weight only"
        ),
    )
    required_group.add_argument(
        "--qembedding_group_size", type=int, required=False, help="Group size for embedding quantization."
    )
    required_group.add_argument(
        "--max_seq_len",
        type=int,
        required=False,
        help="Maximum sequence length for the model. If not specified, uses the model's default max_position_embeddings.",
    )
    required_group.add_argument(
        "--dtype",
        type=str,
        choices=["float32", "float16", "bfloat16"],
        required=False,
        help="Data type for model weights. Options: float32, float16, bfloat16. Default: float32. For quantization (int8/int4), use the --qlinear arguments.",
    )
    required_group.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda", "cuda:0", "cuda:1", "cuda:2", "cuda:3"],
        required=False,
        help="Device to run the model on. Options: cpu, cuda. Default: cpu.",
    )


class ExecuTorchExportCommand(BaseOptimumCLICommand):
    COMMAND = CommandInfo(name="executorch", help="Export models to ExecuTorch.")

    @staticmethod
    def parse_args(parser: "ArgumentParser"):
        return parse_args_executorch(parser)

    def run(self):
        from ...exporters.executorch import main_export

        # Validate int4 packing format can only be used with CUDA devices and 4w quantization
        device = getattr(self.args, "device", None)
        qlinear_packing_format = getattr(self.args, "qlinear_packing_format", None)
        if qlinear_packing_format:
            if not device or not device.startswith("cuda"):
                raise ValueError(
                    "--qlinear_packing_format can only be used when --device is set to CUDA (e.g., 'cuda', 'cuda:0', etc.)"
                )
            if not self.args.qlinear or self.args.qlinear != "4w":
                raise ValueError("--qlinear_packing_format can only be used when --qlinear is set to '4w'")
        qlinear_encoder_packing_format = getattr(self.args, "qlinear_encoder_packing_format", None)
        if qlinear_encoder_packing_format:
            if not device or not device.startswith("cuda"):
                raise ValueError(
                    "--qlinear_encoder_packing_format can only be used when --device is set to CUDA (e.g., 'cuda', 'cuda:0', etc.)"
                )
            if not self.args.qlinear_encoder or self.args.qlinear_encoder != "4w":
                raise ValueError(
                    "--qlinear_encoder_packing_format can only be used when --qlinear_encoder is set to '4w'"
                )

        kwargs = {}
        if self.args.use_custom_sdpa:
            kwargs["use_custom_sdpa"] = self.args.use_custom_sdpa
        if self.args.use_custom_kv_cache:
            kwargs["use_custom_kv_cache"] = self.args.use_custom_kv_cache
        if self.args.disable_dynamic_shapes:
            kwargs["disable_dynamic_shapes"] = self.args.disable_dynamic_shapes
        if self.args.qlinear:
            kwargs["qlinear"] = self.args.qlinear
        if self.args.qlinear_group_size:
            kwargs["qlinear_group_size"] = self.args.qlinear_group_size
        if qlinear_packing_format:
            kwargs["qlinear_packing_format"] = qlinear_packing_format
        if self.args.qlinear_encoder:
            kwargs["qlinear_encoder"] = self.args.qlinear_encoder
        if self.args.qlinear_encoder_group_size:
            kwargs["qlinear_encoder_group_size"] = self.args.qlinear_encoder_group_size
        if qlinear_encoder_packing_format:
            kwargs["qlinear_encoder_packing_format"] = qlinear_encoder_packing_format
        if self.args.qembedding:
            kwargs["qembedding"] = self.args.qembedding
        if self.args.qembedding_group_size:
            kwargs["qembedding_group_size"] = self.args.qembedding_group_size
        if self.args.max_seq_len:
            kwargs["max_seq_len"] = self.args.max_seq_len
        if hasattr(self.args, "dtype") and self.args.dtype:
            kwargs["dtype"] = self.args.dtype
        if hasattr(self.args, "device") and self.args.device:
            kwargs["device"] = self.args.device

        main_export(
            model_name_or_path=self.args.model,
            task=self.args.task,
            recipe=self.args.recipe,
            output_dir=self.args.output_dir,
            **kwargs,
        )
