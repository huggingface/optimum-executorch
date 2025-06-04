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

from typing import Dict, Optional

import torch
from torch.export import ExportedProgram
from torch.nn.attention import SDPBackend
from transformers import (
    AutoProcessor,
    PreTrainedModel,
    StaticCache,
    T5ForConditionalGeneration,
    WhisperForConditionalGeneration,
)
from transformers.generation.configuration_utils import GenerationConfig

from optimum.executorch.attentions.custom_sdpa import get_custom_sdpa_for_ring_kv_cache
from optimum.utils.import_utils import is_transformers_version

from .utils import save_config_to_constant_methods


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
        input_ids=None,
        cache_position=None,
        dynamic_shapes: Optional[dict] = None,
        strict: Optional[bool] = None,
    ) -> Dict[str, ExportedProgram]:
        example_input_ids = input_ids if input_ids is not None else torch.tensor([[1]], dtype=torch.long)
        example_cache_position = cache_position if cache_position is not None else torch.tensor([0], dtype=torch.long)

        if is_transformers_version(">=", "4.52.0.dev0"):
            from transformers.integrations.executorch import (
                TorchExportableModuleForDecoderOnlyLM,
            )

            max_batch_size = 1
            max_cache_len = 4094
            exportable_module = TorchExportableModuleForDecoderOnlyLM(self.model, max_batch_size, max_cache_len)
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
                exported_program = exportable_module.export(example_input_ids, example_cache_position)
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
                    args=(example_input_ids, example_cache_position),
                    kwargs={},
                    dynamic_shapes=dynamic_shapes,
                    strict=strict if strict is not None else True,
                )
        else:
            from transformers.integrations.executorch import (
                convert_and_export_with_cache,
            )

            exported_program = convert_and_export_with_cache(
                self.model, example_input_ids, example_cache_position, dynamic_shapes, strict
            )

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
                max_static_cache_length=self.generation_config.cache_config.max_cache_len,
                batch_size=self.generation_config.cache_config.batch_size,
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
