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
import time
from typing import Dict


class Stats:
    """Python equivalent of the ExecuTorch C++ Stats structure for measuring LLM execution latency.

    This class provides methods to track various timestamps during model execution,
    including model loading, inference, token generation, and sampling times.
    """

    def __init__(self):
        # Scaling factor for timestamps - in ms
        self.SCALING_FACTOR_UNITS_PER_SECOND = 1000

        # Time stamps for different execution stages
        self.model_load_start_ms = 0
        self.model_load_end_ms = 0
        self.inference_start_ms = 0
        self.token_encode_end_ms = 0
        self.model_execution_start_ms = 0
        self.model_execution_end_ms = 0
        self.prompt_eval_end_ms = 0
        self.first_token_ms = 0
        self.inference_end_ms = 0

        # Sampling time tracking
        self.aggregate_sampling_time_ms = 0
        self._aggregate_sampling_timer_start_timestamp = 0

        # Token counts
        self.num_prompt_tokens = 0
        self.num_generated_tokens = 0

    def on_model_load_start(self):
        """Mark the start of model loading."""
        self.model_load_start_ms = self._time_in_ms()

    def on_model_load_end(self):
        """Mark the end of model loading."""
        self.model_load_end_ms = self._time_in_ms()

    def on_inference_start(self):
        """Mark the start of inference (includes tokenizer encode time)."""
        self.inference_start_ms = self._time_in_ms()

    def on_token_encode_end(self):
        """Mark the end of token encoding."""
        self.token_encode_end_ms = self._time_in_ms()

    def on_model_execution_start(self):
        """Mark the start of the model's forward execution."""
        self.model_execution_start_ms = self._time_in_ms()

    def on_model_execution_end(self):
        """Mark the end of the model's forward execution."""
        self.model_execution_end_ms = self._time_in_ms()

    def on_prompt_eval_end(self):
        """Mark the end of prompt evaluation."""
        self.prompt_eval_end_ms = self._time_in_ms()

    def on_first_token(self):
        """Mark when the first generated token is emitted."""
        self.first_token_ms = self._time_in_ms()

    def on_inference_end(self):
        """Mark the end of inference/generation."""
        self.inference_end_ms = self._time_in_ms()

    def on_sampling_begin(self):
        """Mark the start of sampling."""
        self._aggregate_sampling_timer_start_timestamp = self._time_in_ms()

    def on_sampling_end(self):
        """Mark the end of sampling and update the aggregate sampling time."""
        self.aggregate_sampling_time_ms += self._time_in_ms() - self._aggregate_sampling_timer_start_timestamp
        self._aggregate_sampling_timer_start_timestamp = 0

    def set_num_prompt_tokens(self, count: int):
        """Set the number of prompt tokens."""
        self.num_prompt_tokens = count

    def set_num_generated_tokens(self, count: int):
        """Set the number of generated tokens."""
        self.num_generated_tokens = count

    def reset(self, all_stats: bool = False):
        """Reset stats, optionally including model load times."""
        if all_stats:
            self.model_load_start_ms = 0
            self.model_load_end_ms = 0

        self.inference_start_ms = 0
        self.token_encode_end_ms = 0
        self.model_execution_start_ms = 0
        self.model_execution_end_ms = 0
        self.prompt_eval_end_ms = 0
        self.first_token_ms = 0
        self.inference_end_ms = 0
        self.aggregate_sampling_time_ms = 0
        self.num_prompt_tokens = 0
        self.num_generated_tokens = 0
        self._aggregate_sampling_timer_start_timestamp = 0

    def to_json(self) -> Dict:
        """Convert the stats to a JSON-serializable dictionary."""
        return {
            "prompt_tokens": self.num_prompt_tokens,
            "generated_tokens": self.num_generated_tokens,
            "model_load_start_ms": self.model_load_start_ms,
            "model_load_end_ms": self.model_load_end_ms,
            "inference_start_ms": self.inference_start_ms,
            "token_encode_end_ms": self.token_encode_end_ms,
            "model_execution_start_ms": self.model_execution_start_ms,
            "model_execution_end_ms": self.model_execution_end_ms,
            "inference_end_ms": self.inference_end_ms,
            "prompt_eval_end_ms": self.prompt_eval_end_ms,
            "first_token_ms": self.first_token_ms,
            "aggregate_sampling_time_ms": self.aggregate_sampling_time_ms,
            "SCALING_FACTOR_UNITS_PER_SECOND": self.SCALING_FACTOR_UNITS_PER_SECOND,
        }

    def to_json_string(self) -> str:
        """Convert the stats to a JSON string."""
        return json.dumps(self.to_json())

    def print_report(self):
        """Print a report of the stats, similar to the C++ implementation."""
        print(
            "\n⚠️ DISCLAIMER: Python-based perf measurements are approximate and may not "
            "match absolute speeds on Android/iOS apps. They are intended for relative "
            "comparisons—-e.g. SDPA vs. custom SDPA, FP16 vs. FP32—-so you can gauge "
            "performance improvements from each optimization step. For end-to-end, "
            "platform-accurate benchmarks, please use the official ExecuTorch apps:\n"
            "  • iOS:     https://github.com/pytorch/executorch/tree/main/extension/benchmark/apple/Benchmark\n"
            "  • Android: https://github.com/pytorch/executorch/tree/main/extension/benchmark/android/benchmark\n"
        )
        print(f"PyTorchObserver {self.to_json_string()}")

        print(f"\tPrompt Tokens: {self.num_prompt_tokens} Generated Tokens: {self.num_generated_tokens}")

        model_load_time = (self.model_load_end_ms - self.model_load_start_ms) / self.SCALING_FACTOR_UNITS_PER_SECOND
        print(f"\tModel Load Time:\t\t{model_load_time:.6f} (seconds)")

        inference_time_ms = self.inference_end_ms - self.inference_start_ms
        inference_time = inference_time_ms / self.SCALING_FACTOR_UNITS_PER_SECOND

        if inference_time_ms > 0 and self.num_generated_tokens > 0:
            inference_rate = (self.num_generated_tokens / inference_time_ms) * self.SCALING_FACTOR_UNITS_PER_SECOND
        else:
            inference_rate = 0

        print(
            f"\tTotal inference time:\t\t{inference_time:.6f} (seconds)\t\t Rate: \t{inference_rate:.6f} (tokens/second)"
        )

        prompt_eval_time = (self.prompt_eval_end_ms - self.inference_start_ms) / self.SCALING_FACTOR_UNITS_PER_SECOND

        if (self.prompt_eval_end_ms - self.inference_start_ms) > 0 and self.num_prompt_tokens > 0:
            prompt_eval_rate = (
                self.num_prompt_tokens / (self.prompt_eval_end_ms - self.inference_start_ms)
            ) * self.SCALING_FACTOR_UNITS_PER_SECOND
        else:
            prompt_eval_rate = 0

        print(
            f"\t\tPrompt evaluation:\t{prompt_eval_time:.6f} (seconds)\t\t Rate: \t{prompt_eval_rate:.6f} (tokens/second)"
        )

        eval_time = (self.inference_end_ms - self.prompt_eval_end_ms) / self.SCALING_FACTOR_UNITS_PER_SECOND

        if (self.inference_end_ms - self.prompt_eval_end_ms) > 0 and self.num_generated_tokens > 0:
            eval_rate = (
                self.num_generated_tokens / (self.inference_end_ms - self.prompt_eval_end_ms)
            ) * self.SCALING_FACTOR_UNITS_PER_SECOND
        else:
            eval_rate = 0

        print(
            f"\t\tGenerated {self.num_generated_tokens} tokens:\t{eval_time:.6f} (seconds)\t\t Rate: \t{eval_rate:.6f} (tokens/second)"
        )

        time_to_first_token = (self.first_token_ms - self.inference_start_ms) / self.SCALING_FACTOR_UNITS_PER_SECOND
        print(f"\tTime to first generated token:\t{time_to_first_token:.6f} (seconds)")

        sampling_time = self.aggregate_sampling_time_ms / self.SCALING_FACTOR_UNITS_PER_SECOND
        print(
            f"\tSampling time over {self.num_prompt_tokens + self.num_generated_tokens} tokens:\t{sampling_time:.6f} (seconds)"
        )

    def _time_in_ms(self) -> int:
        """Get the current time in milliseconds."""
        return int(time.time() * 1000)
