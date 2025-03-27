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

import torch


def check_close_recursively(eager_outputs, exported_outputs, atol=1e-4, rtol=1e-4):
    is_close = False
    if isinstance(eager_outputs, torch.Tensor):
        torch.testing.assert_close(eager_outputs, exported_outputs, atol=atol, rtol=rtol)
        return True
    elif isinstance(eager_outputs, (tuple, list)):
        for eager_output, exported_output in zip(eager_outputs, exported_outputs):
            is_close = is_close or check_close_recursively(eager_output, exported_output)
        return is_close
    elif isinstance(eager_outputs, dict):
        for key in eager_outputs:
            is_close = is_close or check_close_recursively(eager_outputs[key], exported_outputs[key])
        return is_close
    return is_close
