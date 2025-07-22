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
        "--qlinear",
        type=str,
        choices=["8da4w", "4w", "8w"],
        required=False,
        help=(
            "Quantization config for linear layers.\n\n"
            "Options:\n"
            "  8da4w - 8-bit dynamic activation, 4-bit weight with group_size = 32\n"
            "  4w    - 4-bit weight only, per group with group_size = 32\n"
            "  8w    - 8-bit weight only, per channel"
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
            "  4w    - 4-bit weight only, per group with group_size = 32\n"
            "  8w    - 8-bit weight only, per channel"
        ),
    )


class ExecuTorchExportCommand(BaseOptimumCLICommand):
    COMMAND = CommandInfo(name="executorch", help="Export models to ExecuTorch.")

    @staticmethod
    def parse_args(parser: "ArgumentParser"):
        return parse_args_executorch(parser)

    def run(self):
        from ...exporters.executorch import main_export

        kwargs = {}
        if self.args.use_custom_sdpa:
            kwargs["use_custom_sdpa"] = self.args.use_custom_sdpa
        if self.args.use_custom_kv_cache:
            kwargs["use_custom_kv_cache"] = self.args.use_custom_kv_cache
        if self.args.qlinear:
            kwargs["qlinear"] = self.args.qlinear
        if self.args.qembedding:
            kwargs["qembedding"] = self.args.qembedding

        main_export(
            model_name_or_path=self.args.model,
            task=self.args.task,
            recipe=self.args.recipe,
            output_dir=self.args.output_dir,
            **kwargs,
        )
