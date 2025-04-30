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

from typing import Optional, Tuple, Union

import torch
from executorch.extension.llm.custom_ops.custom_ops import custom_sdpa  # noqa


def custom_sdpa_with_start_pos_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Union[torch.Tensor, "BlockMask"],  # noqa
    scaling: Optional[float] = None,
    softcap: Optional[float] = None,
    head_mask: Optional[torch.Tensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, None]:
    # This is before the transpose
    max_seq_len = key.shape[2]

    # FA2 uses non-transposed inputs
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)

    # Convert the hell out of the inputs to fp32 and back
    input_dtype = query.dtype
    query = query.to(torch.float32)
    key = key.to(torch.float32)
    value = value.to(torch.float32)

    # Ignore the causal flag from kwargs but use the one in module
    kwargs.pop("is_causal", None)

    # Calculate the input pos from attention mask.
    # Branch out for float vs bool mask
    # assert attention_mask.dim() == 2, f"attention_mask must be a 2D matrix."
    attention_mask = attention_mask.reshape(-1, max_seq_len)
    first_row_mask = attention_mask[0, :]
    # [0, 0, 0, 0, -inf, -inf, -inf, -inf], start_pos = 3
    start_pos = torch.argmin(first_row_mask).item() - 1
    output = torch.ops.llama.custom_sdpa(
        query,
        key,
        value,
        start_pos=start_pos,
        attn_mask=None,
        drpout_p=0.0,
        is_causal=module.is_causal,
        scale=scaling,
    )
    return output.to(input_dtype), None
