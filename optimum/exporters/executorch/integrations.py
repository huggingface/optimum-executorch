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
from typing import Dict, Optional

import torch
from packaging.version import parse
from torch.export import ExportedProgram
from torch.nn.attention import SDPBackend
from transformers import (
    AutoProcessor,
    PreTrainedModel,
    StaticCache,
    T5ForConditionalGeneration,
    WhisperForConditionalGeneration,
    VoxtralForConditionalGeneration,
    Gemma3ForConditionalGeneration,
)
from transformers.configuration_utils import PretrainedConfig
from transformers.generation.configuration_utils import GenerationConfig
from transformers.cache_utils import HybridCache
from transformers.integrations.executorch import sdpa_mask_without_vmap
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS

from executorch import version as executorch_version
from optimum.executorch.attentions.custom_sdpa import get_custom_sdpa_for_ring_kv_cache
from optimum.utils.import_utils import is_transformers_version

from .utils import save_config_to_constant_methods

# TODO(JZ): upstream changes here to transformers.
class TorchExportableModuleWithStaticCache(torch.nn.Module):
    """
    A recipe module designed to make a `PreTrainedModel` exportable with `torch.export`,
    specifically for decoder-only LM to `StaticCache`. This module ensures that the
    exported model is compatible with further lowering and execution in `ExecuTorch`.

    Note:
        This class is specifically designed to support export process using `torch.export`
        in a way that ensures the model can be further lowered and run efficiently in `ExecuTorch`.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        config: PretrainedConfig,
        generation_config: GenerationConfig,
    ):
        """
        Initializes the wrapper module with the pretrained model.

        Args:
            model (`PreTrainedModel`): The pretrained model to wrap. The model must have caching
            enabled and use a 'static' caching implementation.

        Raises:
            AssertionError: If the pretrained model does not have caching enabled or if it does
            not use a 'static' caching implementation in `model.generation_config`.
        """
        super().__init__()

        # Sanity checks
        if generation_config is None:
            raise AssertionError(
                "The model must have a generation config to be exported with static caching. "
                "Please set `generation_config`."
            )

        if not generation_config.use_cache:
            raise AssertionError(
                "The model must have caching enabled to be exported with static caching. "
                "Please set `generation_config.use_cache=True`."
            )

        if generation_config.cache_implementation != "static":
            raise AssertionError(
                "The model must use a 'static' caching implementation to be exported with static caching. "
                "Please set `generation_config.cache_implementation='static'`."
            )

        self.model = model
        self.config = config
        self.generation_config = generation_config
        self.static_cache = StaticCache(
            config=config,
            max_batch_size=self.generation_config.cache_config.batch_size,
            max_cache_len=self.generation_config.cache_config.max_cache_len,
            device=self.generation_config.cache_config.device,
            dtype=self.model.dtype,
        )
        # TODO(JZ): figure out why len(self.static_cache) doesn't work like it does in upstream.
        for i in range(len(self.static_cache.key_cache)):
            self.register_buffer(f"key_cache_{i}", self.static_cache.key_cache[i], persistent=False)
            self.register_buffer(f"value_cache_{i}", self.static_cache.value_cache[i], persistent=False)

    def forward(
        self,
        *,
        cache_position: torch.Tensor,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ):
        """
        Forward pass of the module, which is compatible with the ExecuTorch runtime.

        Args:
            input_ids (`torch.Tensor`): Tensor representing current input token id to the module.
            cache_position (`torch.Tensor`): Tensor representing current input position in the cache.

        Returns:
            torch.Tensor: Logits output from the model.

        This forward adapter serves two primary purposes:

        1. **Making the Model `torch.export`-Compatible**:
            The adapter hides unsupported objects, such as the `Cache`, from the graph inputs and outputs,
            enabling the model to be exportable using `torch.export` without encountering issues.

        2. **Ensuring Compatibility with `ExecuTorch` runtime**:
            The adapter matches the model's forward signature with that in `executorch/extension/llm/runner`,
            ensuring that the exported model can be executed in `ExecuTorch` out-of-the-box.
        """
        if input_ids is not None:
            _, seqlen = input_ids.shape
        else:
            _, seqlen, _ = inputs_embeds.shape
        position_ids = cache_position.unsqueeze(0)
        past_key_values = self.static_cache

        outs = self.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=None,
            position_ids=position_ids,
            cache_position=cache_position,
            past_key_values=past_key_values,
            use_cache=True,
        )
        return outs.logits

    @staticmethod
    def generate(
        exported_program: torch.export.ExportedProgram,
        prompt_token_ids: torch.Tensor,
        max_new_tokens: int,
    ) -> torch.Tensor:
        """
        Generate a sequence of tokens using an exported program.

        This util function is designed to test exported models by simulating the generation process.
        It processes the input prompt tokens sequentially (no parallel prefill).
        This generate function is not intended to replace the original `generate` method, and the support
        for leveraging the original `generate` is potentially planned!

        Args:
            exported_program (`torch.export.ExportedProgram`): The exported program generated via `torch.export`.
            prompt_token_ids (`torch.Tensor`): Tensor representing the input prompt token IDs.
            max_new_tokens (`int`): Maximum number of new tokens to generate. Note that the total generation
                length is limited by both `max_new_tokens` and the model's cache size.

        Returns:
            torch.Tensor: A tensor containing the generated sequence of token IDs, including the original prompt tokens.
        """
        device = prompt_token_ids.device
        prompt_token_len = prompt_token_ids.shape[-1]
        max_generation_length = prompt_token_len + max_new_tokens
        for buffer_name, buffer in exported_program.named_buffers():
            if buffer_name.startswith("key_cache"):
                max_cache_len = buffer.shape[2]
                max_generation_length = min(max_generation_length, max_cache_len)
                break

        response_tokens = []
        for input_pos in range(min(max_generation_length, prompt_token_len)):
            result = exported_program.module().forward(
                input_ids=prompt_token_ids[:, input_pos : input_pos + 1],
                cache_position=torch.tensor([input_pos], dtype=torch.long, device=device),
            )
            response_tokens.append(prompt_token_ids[0][input_pos].item())

        current_token = torch.argmax(result[:, -1, :], dim=-1).item()
        response_tokens.append(current_token)

        while len(response_tokens) < max_generation_length:
            result = exported_program.module().forward(
                input_ids=torch.tensor([[current_token]], dtype=torch.long, device=device),
                cache_position=torch.tensor([len(response_tokens)], dtype=torch.long, device=device),
            )
            current_token = torch.argmax(result[:, -1, :], dim=-1).item()
            response_tokens.append(current_token)

        return torch.tensor([response_tokens], dtype=torch.long, device=device)


class TorchExportableModuleWithHybridCache(torch.nn.Module):
    """
    A recipe module designed to make a `PreTrainedModel` exportable with `torch.export`,
    specifically for decoder-only LM to `HybridCache`. This module ensures that the
    exported model is compatible with further lowering and execution in `ExecuTorch`.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        max_batch_size: int = 1,
        max_cache_len: int = 4096,
    ):
        """
        Initializes the exportable module with `HybridCache`.

        Args:
            model (`PreTrainedModel`): The pretrained model to wrap.
            max_batch_size (int): Maximum batch size for the cache.
            max_cache_len (int): Maximum sequence length for the cache.

        Raises:
            AssertionError: If the model doesn't have the expected configuration for HybridCache.
        """
        super().__init__()
        self.model = model

        # Verify the model is configured for HybridCache
        if not self.model.config.text_config.use_cache:
            raise AssertionError("Model must have caching enabled")

        # Initialize the HybridCache
        self.cache = HybridCache(
            config=self.model.config.text_config,
            max_batch_size=max_batch_size,
            max_cache_len=max_cache_len,
            device=self.model.device,
            dtype=self.model.dtype,
        )

        # Register all key and value cache tensors as buffers
        for i in range(len(self.cache.key_cache)):
            self.register_buffer(f"key_cache_{i}", self.cache.key_cache[i], persistent=False)
            self.register_buffer(f"value_cache_{i}", self.cache.value_cache[i], persistent=False)

    def forward(
        self,
        *,
        cache_position: torch.Tensor,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of the module, which is compatible with the ExecuTorch llm runner.

        Args:
            cache_position (`torch.Tensor`): Tensor representing current input position in the cache.
            input_ids (`torch.Tensor`): Tensor representing current input token id to the module.
            inputs_embeds (`torch.Tensor`): Optional tensor representing input embeddings.

        Returns:
            torch.Tensor: Logits output from the model.
        """
        batch_size = input_ids.shape[0] if input_ids is not None else inputs_embeds.shape[0]

        # Generate position_ids from cache_position
        position_ids = cache_position.unsqueeze(0).expand(batch_size, -1)

        # Forward pass with the model
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=None,
            position_ids=position_ids,
            past_key_values=self.cache,
            use_cache=True,
            cache_position=cache_position,
            inputs_embeds=inputs_embeds,
        )

        # Return only the logits to simplify the export
        return outputs.logits


class TorchExportableModuleForDecoderOnlyLM(torch.nn.Module):
    """
    A recipe module designed to make a `PreTrainedModel` exportable with `torch.export`,
    specifically for image-text LM with cache. This module ensures that the
    exported model is compatible with further lowering and execution in `ExecuTorch`.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        config: PretrainedConfig,
        generation_config: GenerationConfig,
        max_batch_size: int = 1,
        max_cache_len: int = 4096,
    ):
        """
        Initializes the exportable module with `HybridCache`.

        Args:
            model (`PreTrainedModel`): The pretrained model to wrap.
            max_batch_size (int): Maximum batch size for the cache.
            max_cache_len (int): Maximum sequence length for the cache.

        Raises:
            ValueError: If the model is configured with a unsupported cache implementation.
        """
        super().__init__()
    
        if not hasattr(config, "use_cache") or config.use_cache is False:
            raise ValueError("The model must have caching enabled to be performant.")

        if hasattr(config, "layer_types") and getattr(config, "sliding_window", None) is not None:
            self.model = TorchExportableModuleWithHybridCache(model, max_batch_size, max_cache_len)
        else:
            # If `layer_types` is not specified explicitly in the config or `sliding_window` is null,
            # there is only 1 type of layers, so export will use `StaticCache` by default.
            logging.info(
                "Using `StaticCache` for export as `layer_types` is not specified or `sliding_window` is `null` in the config."
            )
            self.model = TorchExportableModuleWithStaticCache(model, config, generation_config)
        # This is the same as sdpa, but mask creation does not use `vmap` which is not exportable
        ALL_MASK_ATTENTION_FUNCTIONS.register("sdpa_without_vmap", sdpa_mask_without_vmap)
        ALL_ATTENTION_FUNCTIONS.register("sdpa_without_vmap", ALL_ATTENTION_FUNCTIONS["sdpa"])
        self.model.model.config._attn_implementation = "sdpa_without_vmap"

    def forward(
        self,
        *,
        cache_position: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        input_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of the module, which is compatible with the ExecuTorch llm runner.

        Args:
            input_embeds (`torch.Tensor`): Tensor representing current input embeddings to the module.
            cache_position (`torch.Tensor`): Tensor representing current input position in the cache.

        Returns:
            torch.Tensor: Logits output from the model.
        """
        return self.model.forward(
            cache_position=cache_position,
            input_ids=input_ids,
            inputs_embeds=input_embeds,
        )

    def export(
        self,
        *,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.Tensor] = None,
        dynamic_shapes: Optional[dict] = None,
        strict: Optional[bool] = None,
    ) -> torch.export.ExportedProgram:
        """
        Export the wrapped module using `torch.export`.

        Args:
            input_ids (`Optional[torch.Tensor]`):
                Tensor representing current input token id to the module. If not provided, a default tensor will be used.
            input_embeds (`Optional[torch.Tensor]`):
                Tensor representing current input embeddings to the module. If not provided, a default tensor will be used.
            cache_position (`Optional[torch.Tensor]`):
                Tensor representing current input position in the cache. If not provided, a default tensor will be used.
            dynamic_shapes (`Optional[dict]`):
                Dynamic shapes to use for export if specified.
            strict(`Optional[bool]`):
                Flag to instruct `torch.export` to use `torchdynamo`.
        """
        if hasattr(self.model, "base_model_prefix"):
            base = getattr(self.model, self.model.base_model_prefix, self.model)
            model_device = base.device
        elif hasattr(self.model, "model"):
            model_device = self.model.model.device
        else:
            model_device = "cpu"
            logging.warning(
                "TorchExportableModuleForImageTextLM.export Can't infer device from the model. Set to CPU by default."
            )

        if not ((input_ids is None) ^ (inputs_embeds is None)):
            raise ValueError("Must specify either input_ids or inputs_embeds")

        if input_ids:
            example_input_ids = (
                input_ids if input_ids is not None else torch.tensor([[1]], dtype=torch.long, device=model_device)
            )
            example_cache_position = (
                cache_position if cache_position is not None else torch.tensor([0], dtype=torch.long, device=model_device)
            )
            exported_program = torch.export.export(
                self.model,
                args=(example_input_ids, example_cache_position),
                kwargs={},
                dynamic_shapes=dynamic_shapes,
                strict=strict if strict is not None else True,
            )
        else:
            seq_length = 3
            # TODO(JZ): remove this and pass this in instead?
            # if dynamic_shapes is None:
            #     seq_len_dim = torch.export.Dim("seq_length_dim", max=seq_length)
            #     dynamic_shapes = {
            #         "inputs_embeds": {1: seq_len_dim},
            #         "cache_position": {0: seq_len_dim},
            #     }
            exported_program = torch.export.export(
                self.model,
                args=(),
                kwargs={"cache_position": cache_position, "inputs_embeds": inputs_embeds},
                dynamic_shapes=dynamic_shapes,
                strict=strict if strict is not None else True,
            )

        return exported_program
    

class ImageEncoderExportableModule(torch.nn.Module):
    """
    A wrapper module designed to make a vision encoder-only model exportable with `torch.export`.
    This module ensures that the exported model is compatible with ExecuTorch.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values):
        """
        Projects the last hidden state from the vision model into language model space.

        Args:
            pixel_values (`torch.FloatTensor]` of shape `(batch_size, channels, height, width)`)
               The tensors corresponding to the input images.
        Returns:
            image_features (`torch.Tensor`): Image feature tensor of shape `(num_images, image_length, embed_dim)`).
        """
        vision_outputs = self.model.vision_tower(pixel_values=pixel_values).last_hidden_state
        image_features = self.model.multi_modal_projector(vision_outputs)
        return image_features


class VoxtralEncoderExportableModule(torch.nn.Module):
    """
    Subgraph which handles all of the audio-related work: encoder, multimodal projection, combinining with text tokens.
    The result of this subgraph should stream directly into the decoder subgraph.
    """
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.audio_encoder = model.audio_tower
        self.mm_projector = model.multi_modal_projector
        self.intermediate_size = model.config.audio_config.intermediate_size
        self.audio_token_id = model.config.audio_token_id

    def forward(
        self,
        input_features: torch.FloatTensor,
        inputs_embeds: torch.FloatTensor,
        input_ids: torch.LongTensor,
    ):
        audio_outputs = self.audio_encoder(input_features)
        audio_hidden_states = audio_outputs.last_hidden_state
        audio_hidden_states = audio_hidden_states.reshape(-1, self.intermediate_size)
        audio_embeds = self.mm_projector(audio_hidden_states)

        audio_token_mask = input_ids == self.audio_token_id
        inputs_embeds[audio_token_mask] = audio_embeds
        
        return inputs_embeds
    

class MultiModalTextToTextExportableModule(torch.nn.Module):
    """
    A wrapper module designed to make an multimodal model, e.g. image-text-to-text, exportable with `torch.export`.
    This module ensures that the exported model is compatible with ExecuTorch.
    """

    def __init__(self, model, use_custom_kv_cache=False, use_custom_sdpa=False):
        super().__init__()
        self.model = model
        self.config = model.config
        self.use_custom_kv_cache = use_custom_kv_cache
        self.use_custom_sdpa = use_custom_sdpa
        self.metadata = save_config_to_constant_methods(model.config.text_config, model.generation_config)
        logging.info(f"Metadata to be recorded in PTE: {self.metadata}")

    def _prepare_vision_embedding_export_inputs(self):
        """
        Prepare example inputs and configurations for export.

        Returns:
            pixel_values (torch.Tensor): Example pixel values tensor.
            dynamic_shapes (dict or None): Dynamic shape specifications for export.
            strict (bool): Whether to use strict export mode.
        """
        image_size = self.config.vision_config.image_size
        pixel_values = torch.rand((1, 3, image_size, image_size))
        dynamic_shapes = None
        strict = False

        return pixel_values, dynamic_shapes, strict

    def _prepare_audio_embedding_export_inputs(self):
        # TODO(JZ): specific to Voxtral, should generalize.
        batch_size = 3
        chunk_length = self.model.audio_tower.config.max_source_positions * self.model.audio_tower.conv1.stride[0] * self.model.audio_tower.conv2.stride[0]
        spectrogram_features = 128
        audio_input = torch.rand(batch_size, spectrogram_features, chunk_length)

        max_audio_len = 150  # In s, should be a multiple of 30.
        dynamic_shapes = {
            "input_features": {
                0: torch.export.Dim("batch_size", min=1, max=max_audio_len/30),
                1: torch.export.Dim.STATIC,
                2: torch.export.Dim.STATIC,
            },
        }
        
        strict = True

        return audio_input, dynamic_shapes, strict
    
    def _prepare_text_embedding_export_inputs(self):
        """
        Prepare example inputs and configurations for export.

        Returns:
            input_ids (torch.Tensor): Example input IDs tensor.
            dynamic_shapes (dict or None): Dynamic shape specifications for export.
            strict (bool): Whether to use strict export mode.
        """
        # Prepare inputs with dynamic shapes
        seq_length = 3  # Sequence length > 1 to avoid specialization issues
        example_input_ids = torch.zeros((1, seq_length), dtype=torch.long)
        max_seq_len = self.metadata.get("get_max_seq_len")
        sliding_window = self.metadata.get("sliding_window", float("inf"))
        max_dim = min(max_seq_len, sliding_window) - 1
        seq_len_dim = torch.export.Dim("seq_length_dim", max=max_dim)
        dynamic_shapes = {
            "input": {1: seq_len_dim},
        }  # nn.embedding forward() here - https://github.com/pytorch/pytorch/blob/febf3c475e6fe369b41ef009f3598659a6df0911/torch/nn/modules/sparse.py#L15.
        strict = parse(torch.__version__) != parse("2.7.0")  # Workaround for PyTorch bug #150994
        return example_input_ids, dynamic_shapes, strict

    def _prepare_decoder_only_export_inputs(self):
        """
        Prepare example inputs and configurations for export.

        Returns:
            inputs_embeds (torch.Tensor): Example input embeddings tensor.
            cache_position (torch.Tensor): Example cache position tensor.
            dynamic_shapes (dict or None): Dynamic shape specifications for export.
            strict (bool): Whether to use strict export mode.
        """

        # Prepare inputs with dynamic shapes
        seq_length = 3
        example_inputs_embeds = torch.zeros((1, seq_length, self.config.text_config.hidden_size), dtype=torch.float)
        example_cache_position = torch.arange(seq_length, dtype=torch.long)
        max_seq_len = self.metadata.get("get_max_seq_len")
        sliding_window = self.metadata.get("sliding_window", float("inf"))
        max_dim = min(max_seq_len, sliding_window) - 1
        seq_len_dim = torch.export.Dim("seq_length_dim", max=max_dim)
        dynamic_shapes = {
            "inputs_embeds": {1: seq_len_dim},
            "cache_position": {0: seq_len_dim},
        }
        strict = parse(torch.__version__) != parse("2.7.0")  # Workaround for PyTorch bug #150994
        return example_inputs_embeds, example_cache_position, dynamic_shapes, strict


    def _register_attention_mask_for_4_53(self, exportable_module: torch.nn.Module):
        if is_transformers_version(">=", "4.53.0.dev0"):
            from transformers.integrations.executorch import sdpa_mask_without_vmap
            from transformers.masking_utils import AttentionMaskInterface
            from transformers.modeling_utils import AttentionInterface

            _custom_sdpa_for_ring_kv_cache = get_custom_sdpa_for_ring_kv_cache(exportable_module)
            if self.use_custom_sdpa:
                if self.use_custom_kv_cache:
                    AttentionInterface.register("custom_sdpa_ring_kv_cache", _custom_sdpa_for_ring_kv_cache)
                    AttentionMaskInterface.register("custom_sdpa_ring_kv_cache", sdpa_mask_without_vmap)
                    # Manually set the attention implementation to custom_sdpa_ring_kv_cache
                    # This handles both regular sdpa and one for sliding window/local attention
                    exportable_module.model.model.config._attn_implementation = "custom_sdpa_ring_kv_cache"
                else:
                    # Manually set the attention implementation to custom_sdpa_ring_kv_cache
                    # This handles both regular sdpa and one for sliding window/local attention
                    exportable_module.model.model.config._attn_implementation = "custom_sdpa"

    def export(
        self,
    ) -> Dict[str, ExportedProgram]:
        with torch.no_grad():
            # 1. Export text decoder.
            exportable_module = TorchExportableModuleForDecoderOnlyLM(
                self.model.language_model,
                self.config.text_config,
                self.model.generation_config,
                max_batch_size=1,
                max_cache_len=self.metadata.get("get_max_seq_len"),
            )
            exported_programs = {}

            # Custom SDPA for text decoder.
            self._register_attention_mask_for_4_53(exportable_module)

            if self.use_custom_kv_cache:
                from optimum.executorch.attentions.custom_kv_cache import (
                    replace_with_et_custom_kv_cache,
                )

                replace_with_et_custom_kv_cache(
                    exportable_module.model,
                    self.model.config.text_config,
                    self.model.generation_config,
                    self.model.dtype,
                )

            inputs_embeds, cache_position, dynamic_shapes, strict = self._prepare_decoder_only_export_inputs()
            logging.info(
                f"Exporting decoder using inputs_embeds({inputs_embeds.shape}), cache_position({cache_position.shape})={cache_position}, dynamic_shapes={dynamic_shapes}, strict={strict}"
            )
            exported_program = exportable_module.export(
                inputs_embeds=inputs_embeds,
                cache_position=cache_position,
                dynamic_shapes=dynamic_shapes,
                strict=strict
            )
            # Apply RemoveTransposes pass to remove
            # any back-to-back transpose ops that are not needed
            # e.g. output of update_cache is transposed and
            # input to custom_sdpa is transposed.
            from executorch.extension.llm.export.export_passes import (
                RemoveRedundantTransposes,
            )

            mutated_gm = RemoveRedundantTransposes()(exported_program.module())[0]
            exported_program = torch.export.export(
                mutated_gm,
                args=(),
                kwargs={"cache_position": cache_position, "inputs_embeds": inputs_embeds},
                dynamic_shapes=dynamic_shapes,
                strict=strict,
            )
            exported_programs["decoder"] = exported_program

            # 2. Export token embeddings
            input_ids, dynamic_shapes, strict = self._prepare_text_embedding_export_inputs()
            logging.info(f"Exporting token embeddings using input_ids({input_ids.shape}), dynamic_shapes={dynamic_shapes}, strict={strict}")

            token_embeddings_exported_program = torch.export.export(
                self.model.language_model.get_input_embeddings(),
                args=(input_ids,),
                kwargs={},
                dynamic_shapes=dynamic_shapes,
                strict=strict,
            )
            exported_programs["token_embeddings"] = token_embeddings_exported_program

            # 3. Export encoder.
            input_ids = torch.zeros_like(inputs_embeds[:, :, 0], dtype=torch.long)
            input_ids[0, 1] = self.config.audio_token_id  # Make sure we don't have an all-false mask for the imput_embeds.
            if isinstance(self.model, VoxtralForConditionalGeneration):
                # TODO(JZ): specific to Voxtral, should generalize.
                chunk_length = self.model.audio_tower.config.max_source_positions * self.model.audio_tower.conv1.stride[0] * self.model.audio_tower.conv2.stride[0]
                encoder_input_kwargs = {
                    "input_features": torch.rand(3, 128, chunk_length),  # (bsz, features, seq_len)
                    "inputs_embeds": inputs_embeds,
                    "input_ids": input_ids,
                }

                max_audio_len = 150  # In s, should be a multiple of 30. TODO(JZ): make this configurable top-level.
                max_seq_len = self.metadata.get("get_max_seq_len")
                dynamic_shapes = {
                    "input_features": {
                        0: torch.export.Dim("enc_batch_size_dim", min=1, max=max_audio_len//30),
                        # 1: torch.export.Dim.STATIC,
                        # 2: torch.export.Dim.STATIC,
                    },
                    "inputs_embeds": {1: torch.export.Dim("input_embeds_seq_length_dim", max=max_seq_len)},
                    "input_ids": {1: torch.export.Dim("input_ids_seq_length_dim", max=max_seq_len)},
                }

                # self.model.audio_tower.config._attn_implementation = "sdpa_without_vmap"
                self.model.audio_tower.config._attn_implementation = "custom_sdpa"
                audio_encoder = VoxtralEncoderExportableModule(self.model)
                audio_encoder_exported_program = torch.export.export(
                    audio_encoder,
                    args=(),
                    kwargs=encoder_input_kwargs,
                    dynamic_shapes=dynamic_shapes,
                    strict=False,
                )
                exported_programs["audio_encoder"] = audio_encoder_exported_program
            elif isinstance(self.model, Gemma3ForConditionalGeneration):
                pixel_values, dynamic_shapes, strict = self._prepare_vision_embedding_export_inputs()
                logging.info(f"Exporting vision embeddings using pixel_values({pixel_values.shape}), dynamic_shapes={dynamic_shapes}, strict={strict}")
                # Setting the _attn_implementation to "sdpa_without_vmap" for vision encoder
                exportable_module.model.model.vision_tower.config._attn_implementation = "sdpa_without_vmap"
                vision_encoder = ImageEncoderExportableModule(exportable_module.model.model)
                vision_embeddings_exported_program = torch.export.export(
                    vision_encoder,
                    args=(pixel_values,),
                    kwargs={},
                    dynamic_shapes=dynamic_shapes,
                    strict=strict,
                )
                exported_programs["vision_encoder"] = vision_embeddings_exported_program

        return exported_programs
    

class CausalLMExportableModule(torch.nn.Module):
    """
    A wrapper module designed to make a Causal LM model exportable with `torch.export`.
    This module ensures that the exported model is compatible with ExecuTorch.
    """

    def __init__(self, model, use_custom_kv_cache=False, use_custom_sdpa=False):
        super().__init__()
        self.model = model
        self.config = model.config
        self.use_custom_kv_cache = use_custom_kv_cache
        self.use_custom_sdpa = use_custom_sdpa
        self.metadata = save_config_to_constant_methods(model.config, model.generation_config)
        logging.info(f"Metadata to be recorded in PTE: {self.metadata}")

    def _prepare_export_inputs(self):
        """
        Prepare example inputs and configurations for export.

        Returns:
            example_input_ids (torch.Tensor): Example input IDs tensor.
            example_cache_position (torch.Tensor): Example cache position tensor.
            dynamic_shapes (dict or None): Dynamic shape specifications for export.
            strict (bool): Whether to use strict export mode.
        """
        # Default values for legacy or fallback cases
        example_input_ids = torch.tensor([[1]], dtype=torch.long)
        example_cache_position = torch.tensor([0], dtype=torch.long)
        dynamic_shapes = None
        strict = True

        is_using_hybrid_cache_wo_custom_sdpa_kv_cache = (
            hasattr(self.config, "layer_types")
            and getattr(self.config, "sliding_window", None) is not None
            and not (self.use_custom_kv_cache and self.use_custom_sdpa)
        )

        if is_transformers_version(">", "4.52.0") and not is_using_hybrid_cache_wo_custom_sdpa_kv_cache:
            # Prepare inputs with dynamic shapes
            seq_length = 3  # Sequence length > 1 to avoid specialization issues
            example_input_ids = torch.zeros((1, seq_length), dtype=torch.long)
            example_cache_position = torch.arange(seq_length, dtype=torch.long)
            max_seq_len = self.metadata.get("get_max_seq_len")
            sliding_window = self.metadata.get("sliding_window", float("inf"))
            max_dim = min(max_seq_len, sliding_window) - 1
            seq_len_dim = torch.export.Dim("seq_length_dim", max=max_dim)
            dynamic_shapes = {
                "input_ids": {1: seq_len_dim},
                "cache_position": {0: seq_len_dim},
            }
            strict = parse(torch.__version__) != parse("2.7.0")  # Workaround for PyTorch bug #150994

        return example_input_ids, example_cache_position, dynamic_shapes, strict

    def _register_attention_mask_for_4_53(self, exportable_module: torch.nn.Module):
        if is_transformers_version(">=", "4.53.0.dev0"):
            from transformers.integrations.executorch import sdpa_mask_without_vmap
            from transformers.masking_utils import AttentionMaskInterface
            from transformers.modeling_utils import AttentionInterface

            _custom_sdpa_for_ring_kv_cache = get_custom_sdpa_for_ring_kv_cache(exportable_module)
            if self.use_custom_sdpa:
                if self.use_custom_kv_cache:
                    AttentionInterface.register("custom_sdpa_ring_kv_cache", _custom_sdpa_for_ring_kv_cache)
                    AttentionMaskInterface.register("custom_sdpa_ring_kv_cache", sdpa_mask_without_vmap)
                    # Manually set the attention implementation to custom_sdpa_ring_kv_cache
                    # This handles both regular sdpa and one for sliding window/local attention
                    exportable_module.model.model.config._attn_implementation = "custom_sdpa_ring_kv_cache"
                else:
                    # Manually set the attention implementation to custom_sdpa_ring_kv_cache
                    # This handles both regular sdpa and one for sliding window/local attention
                    exportable_module.model.model.config._attn_implementation = "custom_sdpa"

    def export(
        self,
    ) -> Dict[str, ExportedProgram]:
        input_ids, cache_position, dynamic_shapes, strict = self._prepare_export_inputs()
        logging.info(
            f"Exporting using input_ids({input_ids.shape})={input_ids}, cache_position({cache_position.shape})={cache_position}, dynamic_shapes={dynamic_shapes}, strict={strict}"
        )
        if is_transformers_version(">", "4.52.0"):
            from transformers.integrations.executorch import (
                TorchExportableModuleForDecoderOnlyLM,
            )

            exportable_module = TorchExportableModuleForDecoderOnlyLM(
                self.model,
                max_batch_size=1,
                max_cache_len=self.metadata.get("get_max_seq_len"),
            )
            self._register_attention_mask_for_4_53(exportable_module)

            if self.use_custom_kv_cache:
                from optimum.executorch.attentions.custom_kv_cache import (
                    replace_with_et_custom_kv_cache,
                )

                replace_with_et_custom_kv_cache(
                    exportable_module.model,
                    self.model.config,
                    self.model.generation_config,
                    self.model.dtype,
                )

            with torch.no_grad():
                exported_program = exportable_module.export(input_ids, cache_position, dynamic_shapes, strict)
                # Apply RemoveTransposes pass to remove
                # any back-to-back transpose ops that are not needed
                # e.g. output of update_cache is transposed and
                # input to custom_sdpa is transposed.
                from executorch.extension.llm.export.export_passes import (
                    RemoveRedundantTransposes,
                )

                mutated_gm = RemoveRedundantTransposes()(exported_program.module())[0]
                exported_program = torch.export.export(
                    mutated_gm,
                    args=(input_ids, cache_position),
                    kwargs={},
                    dynamic_shapes=dynamic_shapes,
                    strict=strict,
                )
        else:
            # Path to use legacy API, static export only due to pinned transformers version
            from transformers.integrations.executorch import (
                convert_and_export_with_cache,
            )

            exported_program = convert_and_export_with_cache(self.model, input_ids, cache_position)

        return {"model": exported_program}


class VisionEncoderExportableModule(torch.nn.Module):
    """
    A wrapper module designed to make a vision encoder-only model exportable with `torch.export`.
    This module ensures that the exported model is compatible with ExecuTorch.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.config = model.config
        # Metadata to be recorded in the pte model file
        self.metadata = save_config_to_constant_methods(model.config, model.generation_config)

    def forward(self, pixel_values):
        print(f"DEBUG: pixel_values: {pixel_values.shape}")
        print(f"DEBUG: forward: {self.model.method_meta('forward')}")
        return self.model(pixel_values=pixel_values)

    def export(self, pixel_values=None) -> Dict[str, ExportedProgram]:
        if pixel_values is None:
            batch_size = 1
            num_channels = self.config.num_channels
            height = self.config.image_size
            width = self.config.image_size
            pixel_values = torch.rand(batch_size, num_channels, height, width)

        with torch.no_grad():
            return {
                "model": torch.export.export(
                    self.model,
                    args=(),
                    kwargs={"pixel_values": pixel_values},
                    strict=False,
                )
            }


class MaskedLMExportableModule(torch.nn.Module):
    """
    A wrapper module designed to make a Masked LM model exportable with `torch.export`.
    This module ensures that the exported model is compatible with ExecuTorch.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.config = model.config
        # Metadata to be recorded in the pte model file
        self.metadata = save_config_to_constant_methods(model.config, model.generation_config)

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids, attention_mask)

    def export(self, input_ids=None, attention_mask=None) -> Dict[str, ExportedProgram]:
        max_position_embeddings = getattr(self.model.config, "max_position_embeddings", 64)
        max_seq_length = max(max_position_embeddings - 1, 1)
        # Create dummy inputs with expected shapes
        batch_size = 1
        seq_length = max_seq_length
        vocab_size = self.model.config.vocab_size

        # Create example inputs (no need for tokenizer)
        dummy_input_ids = (
            torch.randint(0, vocab_size, (batch_size, seq_length), dtype=torch.long)
            if input_ids is None
            else input_ids
        )
        dummy_attention_mask = (
            torch.ones((batch_size, seq_length), dtype=torch.long) if attention_mask is None else attention_mask
        )

        # Define dynamic shapes with Dim objects, always use Auto
        dynamic_shapes = {
            "input_ids": {1: torch.export.Dim.AUTO},
            "attention_mask": {1: torch.export.Dim.AUTO},
        }

        # Export the model with dynamic dimensions
        with torch.no_grad():
            return {
                "model": torch.export.export(
                    self.model,
                    args=(dummy_input_ids,),
                    kwargs={"attention_mask": dummy_attention_mask},
                    dynamic_shapes=dynamic_shapes,
                    strict=True,
                )
            }


class Seq2SeqLMEncoderExportableModule(torch.nn.Module):
    """
    A wrapper module designed to make a Seq2Seq LM encoder exportable with `torch.export`.
    This module ensures that the exported encoder model is compatible with ExecuTorch.
    """

    def __init__(self, encoder_model):
        super().__init__()
        self.encoder = encoder_model
        self.config = encoder_model.config

    def forward(self, input_ids):
        return self.encoder(input_ids).last_hidden_state


class Seq2SeqLMDecoderExportableModuleWithStaticCache(torch.nn.Module):
    """
    A wrapper module designed to make a Seq2Seq LM decoder exportable with `torch.export`,
    specifically for use with static caching. This module ensures the exported decoder
    is compatible with ExecuTorch.
    """

    def __init__(self, model, max_static_cache_length, batch_size):
        super().__init__()

        # Get the decoder component
        self.decoder = model.get_decoder()
        if isinstance(model, WhisperForConditionalGeneration):
            self.proj_out = model.proj_out
        else:
            self.proj_out = model.lm_head
        self.config = model.config

        # Initialize static cache
        self.static_cache = StaticCache(
            config=self.config,
            max_batch_size=batch_size,
            max_cache_len=max_static_cache_length,
            device="cpu",
            dtype=torch.float32,
        )

        # Register cache buffers to make them exportable
        for i in range(len(self.static_cache.key_cache)):
            self.register_buffer(f"key_cache_{i}", self.static_cache.key_cache[i], persistent=False)
            self.register_buffer(f"value_cache_{i}", self.static_cache.value_cache[i], persistent=False)

    def forward(self, decoder_input_ids, encoder_hidden_states, cache_position):
        # Get outputs from decoder
        outputs = self.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_hidden_states,
            past_key_values=self.static_cache,
            use_cache=True,
            cache_position=cache_position,
        )

        # Apply linear projection (lm head) to obtain logits
        logits = self.proj_out(outputs[0])
        return logits


class Seq2SeqLMExportableModule(torch.nn.Module):
    def __init__(
        self,
        model: PreTrainedModel,
        batch_size=1,
        max_hidden_seq_length=4096,
        cache_implementation="static",
        max_cache_length=1024,
    ):
        super().__init__()

        self.full_model = model
        self.encoder = model.get_encoder()
        self.config = model.config
        self.max_hidden_seq_length = max_hidden_seq_length
        self.generation_config = GenerationConfig(
            use_cache=True,
            max_length=max_cache_length,
            cache_implementation=cache_implementation,
            cache_config={
                "batch_size": batch_size,
                "max_cache_len": max_cache_length,
            },
        )
        if isinstance(self.full_model, WhisperForConditionalGeneration):
            self._processor = AutoProcessor.from_pretrained(model.config._name_or_path)
            self._expected_encoder_input_shape = torch.Size(
                (
                    1,
                    self._processor.feature_extractor.feature_size,
                    self._processor.feature_extractor.nb_max_frames,
                )
            )
        additional_configs = {}
        additional_configs["max_hidden_seq_length"] = max_hidden_seq_length
        # Metadata to be recorded in the pte model file
        self.metadata = save_config_to_constant_methods(
            self.config,
            self.generation_config,
            **additional_configs,
        )
        self.exported_encoder = None
        self.exported_decoder = None

    def _export_encoder(self, encoder_input_ids):
        wrapped_encoder = Seq2SeqLMEncoderExportableModule(self.encoder).to("cpu").eval()

        # Define dynamic sequence length for encoder
        if isinstance(self.full_model, WhisperForConditionalGeneration):
            assert (
                encoder_input_ids.shape == self._expected_encoder_input_shape
            ), f"""This version of Whisper only accepts encoder input of shape {self._expected_encoder_input_shape}, passed shape: {encoder_input_ids.shape}.
                For more infromation, please refer to the Whisper preprocessor config."""
            dynamic_shapes = None
        elif isinstance(self.full_model, T5ForConditionalGeneration):
            encoder_seq_len_dim = torch.export.Dim("encoder_hidden_seq_length", max=self.max_hidden_seq_length)
            dynamic_shapes = {"input_ids": {1: encoder_seq_len_dim}}
        else:
            raise ValueError(
                f"Unsupported model type {type(self.full_model)} for Seq2SeqLMExportableModule encoder export."
            )

        # Export the encoder
        with torch.no_grad():
            exported_encoder = torch.export.export(
                wrapped_encoder,
                (encoder_input_ids,),
                dynamic_shapes=dynamic_shapes,
                strict=True,
            )
        return exported_encoder

    def _export_decoder(self, decoder_input_ids, encoder_hidden_states, cache_position):
        wrapped_decoder = (
            Seq2SeqLMDecoderExportableModuleWithStaticCache(
                model=self.full_model,
                max_static_cache_length=self.generation_config.cache_config.get("max_cache_len"),
                batch_size=self.generation_config.cache_config.get("batch_size"),
            )
            .to("cpu")
            .eval()
        )

        if isinstance(self.full_model, WhisperForConditionalGeneration):
            dynamic_shapes = None
        elif isinstance(self.full_model, T5ForConditionalGeneration):
            # Define dynamic dimension for encoder output sequence length
            encoder_seq_len_dim = torch.export.Dim("encoder_hidden_seq_length", max=self.max_hidden_seq_length)
            dynamic_shapes = {
                "decoder_input_ids": None,
                "encoder_hidden_states": {1: encoder_seq_len_dim},
                "cache_position": None,
            }
        else:
            raise ValueError(
                f"Unsupported model type {type(self.full_model)} for Seq2SeqLMExportableModule decoder export."
            )

        # Export the decoder
        with torch.nn.attention.sdpa_kernel([SDPBackend.MATH]), torch.no_grad():
            exported_decoder = torch.export.export(
                wrapped_decoder,
                (decoder_input_ids, encoder_hidden_states, cache_position),
                dynamic_shapes=dynamic_shapes,
                strict=True,
            )

        return exported_decoder

    def export(
        self,
        encoder_input_ids=None,
        decoder_input_ids=None,
        encoder_hidden_states=None,
        cache_position=None,
    ) -> Dict[str, ExportedProgram]:
        if encoder_input_ids is None:
            if isinstance(self.full_model, WhisperForConditionalGeneration):
                example_encoder_input_ids = torch.rand(self._expected_encoder_input_shape)
            else:
                example_encoder_input_ids = torch.ones((1, 10), dtype=torch.long)
        else:
            example_encoder_input_ids = encoder_input_ids

        self.exported_encoder = self._export_encoder(example_encoder_input_ids)

        if not encoder_hidden_states:
            example_encoder_hidden_states = self.exported_encoder.module()(example_encoder_input_ids)
        else:
            example_encoder_hidden_states = encoder_hidden_states

        example_decoder_input_ids = (
            decoder_input_ids if decoder_input_ids is not None else torch.tensor([[0]], dtype=torch.long)
        )
        example_cache_position = cache_position if cache_position is not None else torch.tensor([0], dtype=torch.long)

        self.exported_decoder = self._export_decoder(
            example_decoder_input_ids,
            example_encoder_hidden_states,
            example_cache_position,
        )

        return {
            "encoder": self.exported_encoder,
            "decoder": self.exported_decoder,
        }

    def generate(self, prompt_token_ids, max_new_tokens):
        with torch.no_grad():
            # Run encoder
            encoder_output = self.exported_encoder.module()(prompt_token_ids)

            # Initialize with start token (0 for T5)
            decoder_input_ids = torch.tensor([[0]], dtype=torch.long)
            generated_ids = [0]

            # Generate tokens one by one
            for i in range(max_new_tokens - 1):
                # Run decoder for next token prediction
                logits = self.exported_decoder.module()(
                    decoder_input_ids,
                    encoder_output,
                    torch.tensor([i], dtype=torch.long),
                )

                # Get next token
                next_token = torch.argmax(logits[:, -1, :], dim=-1).item()
                generated_ids.append(next_token)

                # Update input for next iteration
                decoder_input_ids = torch.tensor([[next_token]], dtype=torch.long)

                # Check if EOS token
                if next_token == self.config.eos_token_id:
                    break

            return generated_ids
