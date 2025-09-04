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
from typing import Dict

import torch
from packaging.version import parse
from torch.export import ExportedProgram
from torch.nn.attention import SDPBackend
from transformers import (
    AutoProcessor,
    PreTrainedModel,
    StaticCache,
    T5ForConditionalGeneration,
    VoxtralForConditionalGeneration,
    WhisperForConditionalGeneration,
)
from transformers.generation.configuration_utils import GenerationConfig
from transformers.integrations.executorch import TorchExportableModuleForDecoderOnlyLM, sdpa_mask_without_vmap
from transformers.masking_utils import AttentionMaskInterface
from transformers.modeling_utils import AttentionInterface

from optimum.executorch.attentions.custom_sdpa import get_custom_sdpa_for_ring_kv_cache

from .utils import save_config_to_constant_methods


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
        self.metadata = save_config_to_constant_methods(model.config.text_config, model.generation_config)
        self.audio_config = model.config.audio_config

    def prepare_export_inputs(self):
        # https://github.com/huggingface/transformers/blob/main/src/transformers/models/voxtral/modeling_voxtral.py#L342
        chunk_length = (
            self.audio_encoder.config.max_source_positions
            * self.audio_encoder.conv1.stride[0]
            * self.audio_encoder.conv2.stride[0]
        )
        input_features = torch.rand(3, self.audio_config.num_mel_bins, chunk_length)  # 3 is an arbitrary batch size.

        max_audio_len = 120  # In s, should be a multiple of 30, see  https://github.com/huggingface/transformers/blob/fbeaf96f9e2291c21277ac658a33ea8752728bf3/src/transformers/models/voxtral/processing_voxtral.py#L93. # TODO(#127): Add CLI args for max_seq_len and max_audio_len.
        dynamic_shapes = {
            "input_features": {
                0: torch.export.Dim("enc_batch_size_dim", min=1, max=max_audio_len // 30),
            },
        }

        return input_features, dynamic_shapes

    def forward(
        self,
        input_features: torch.FloatTensor,
    ):
        """
        Forward pass of the Voxtral encoder module.

        Args:
            input_features (torch.FloatTensor): Raw audio features with shape (batch_size, features, seq_len).

        Returns:
            torch.FloatTensor: Combined embeddings with audio tokens replaced by audio embeddings.
        """
        audio_outputs = self.audio_encoder(input_features)
        audio_hidden_states = audio_outputs.last_hidden_state
        audio_hidden_states = audio_hidden_states.reshape(-1, self.intermediate_size)
        audio_embeds = self.mm_projector(audio_hidden_states)
        # TODO: Remove unsqueeze(0) and do in custom graph pass instead safter removing Voxtral specific code.
        return audio_embeds.unsqueeze(0)  # (1, audio_embed_len, hidden_dim)


class MultiModalTextToTextExportableModule(torch.nn.Module):
    """
    A wrapper module for exporting an early fusion multimodal model (image/audio to text) with `torch.export`.
    This module also ensures that the exported model is compatible with ExecuTorch.

    The module separates the model into three exportable components:
    1. Token embeddings layer for text encoding
    2. Multimodal encoder (audio/vision) for processing non-text inputs
    3. Text decoder for autoregressive generation

    Args:
        model (torch.nn.Module): The multimodal model to export.
        modality (str): The input modality type ("audio" or "vision").
        encoder_name (str): Name of the encoder attribute in the model.
        processor_config (dict, optional): Preprocessor configuration loaded from preprocessor_config.json.
        use_custom_kv_cache (bool): Whether to use custom key-value caching for optimization.
        use_custom_sdpa (bool): Whether to use custom scaled dot-product attention.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        modality: str,
        decoder_name: str,
        encoder_name: str,
        processor_config: dict = None,
        use_custom_kv_cache: bool = False,
        use_custom_sdpa: bool = False,
    ):
        super().__init__()

        if modality not in encoder_name:
            raise ValueError(f'encoder_name "{encoder_name}" does not match specified modality "{modality}".')
        if not hasattr(model, decoder_name):
            raise ValueError(f'Model does not contain decoder "{decoder_name}".')
        if not hasattr(model, encoder_name):
            raise ValueError(f'Model does not contain encoder "{encoder_name}".')

        self.model = model
        self.config = model.config
        self.modality = modality
        self.decoder_name = decoder_name
        self.encoder_name = encoder_name
        self.processor_config = processor_config
        self.use_custom_kv_cache = use_custom_kv_cache
        self.use_custom_sdpa = use_custom_sdpa
        modality_token_placeholder_id_kwargs = {f"{modality}_token_id": getattr(self.config, f"{modality}_token_id")}
        self.metadata = save_config_to_constant_methods(
            model.config.text_config, model.generation_config, processor_config, **modality_token_placeholder_id_kwargs
        )
        logging.info(f"Metadata to be recorded in PTE: {self.metadata}")

    def _prepare_text_embedding_export_inputs(self, max_seq_len: int):
        """
        Prepare example inputs and configurations for export.

        Returns:
            input_ids (torch.Tensor): Example input IDs tensor.
            dynamic_shapes (dict or None): Dynamic shape specifications for export.
            strict (bool): Whether to use strict export mode.
        """
        seq_length = 3  # Sequence length > 1 to avoid specialization issues
        example_input_ids = torch.zeros((1, seq_length), dtype=torch.long)

        seq_len_dim = torch.export.Dim("seq_length_dim", max=max_seq_len)
        dynamic_shapes = {
            "input": {1: seq_len_dim},
        }  # nn.embedding forward() args are here - https://github.com/pytorch/pytorch/blob/febf3c475e6fe369b41ef009f3598659a6df0911/torch/nn/modules/sparse.py#L15.

        return example_input_ids, dynamic_shapes

    def _prepare_decoder_only_export_inputs(self, max_seq_len: int):
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

        seq_len_dim = torch.export.Dim("seq_length_dim", max=max_seq_len)
        dynamic_shapes = {
            "inputs_embeds": {1: seq_len_dim},
            "cache_position": {0: seq_len_dim},
        }

        return example_inputs_embeds, example_cache_position, dynamic_shapes

    def _register_custom_attention(self, exportable_module: torch.nn.Module):
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
        """
        Export the multimodal model into separate ExecuTorch programs.

        Returns:
            Dict[str, ExportedProgram]: Dictionary containing exported programs:
                - "text_decoder": Text generation decoder
                - "token_embedding": Token embedding layer
                - "{modality}_encoder": Multimodal encoder (e.g., "audio_encoder")
        """
        with torch.no_grad():
            max_seq_len = self.metadata.get("get_max_seq_len")
            sliding_window_len = self.metadata.get("sliding_window", float("inf"))
            max_seq_len = min(max_seq_len, sliding_window_len) - 1
            if max_seq_len == sliding_window_len - 1:
                logging.info("Using sliding window as max sequence length in export.")

            # 1. Export text decoder.
            exportable_module = TorchExportableModuleForDecoderOnlyLM(
                getattr(self.model, self.decoder_name),
            )
            exported_programs = {}

            # Custom SDPA for text decoder.
            self._register_custom_attention(exportable_module)

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

            inputs_embeds, cache_position, dynamic_shapes = self._prepare_decoder_only_export_inputs(max_seq_len)
            logging.info(
                f"Exporting decoder using inputs_embeds({inputs_embeds.shape}), cache_position({cache_position.shape})={cache_position}, dynamic_shapes={dynamic_shapes}"
            )
            exported_program = exportable_module.export(
                inputs_embeds=inputs_embeds,
                cache_position=cache_position,
                dynamic_shapes=dynamic_shapes,
                strict=True,
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
                strict=True,
            )
            exported_programs["text_decoder"] = exported_program

            # 2. Export token embeddings
            input_ids, dynamic_shapes = self._prepare_text_embedding_export_inputs(max_seq_len)
            logging.info(
                f"Exporting token embeddings using input_ids({input_ids.shape}), dynamic_shapes={dynamic_shapes}"
            )

            token_embedding_exported_program = torch.export.export(
                getattr(self.model, self.decoder_name).get_input_embeddings(),
                args=(input_ids,),
                kwargs={},
                dynamic_shapes=dynamic_shapes,
                strict=True,
            )
            exported_programs["token_embedding"] = token_embedding_exported_program

            # 3. Export encoder.
            if self.use_custom_sdpa:
                getattr(self.model, self.encoder_name).config._attn_implementation = "custom_sdpa"

            if isinstance(self.model, VoxtralForConditionalGeneration):
                encoder = VoxtralEncoderExportableModule(self.model)
                input_features, dynamic_shapes = encoder.prepare_export_inputs()
            else:
                raise ValueError(f'Multimodal model type "{type(self.model)}" has not been enabled yet for Optimum.')

            encoder_input_kwargs = {
                "input_features": input_features,
            }
            encoder_exported_program = torch.export.export(
                encoder,
                args=(),
                kwargs=encoder_input_kwargs,
                dynamic_shapes=dynamic_shapes,
                strict=True,
            )
            exported_programs[f"{self.modality}_encoder"] = encoder_exported_program

        return exported_programs


class CausalLMExportableModule(torch.nn.Module):
    """
    A wrapper module designed to make a Causal LM model exportable with `torch.export`.
    This module ensures that the exported model is compatible with ExecuTorch.
    """

    def __init__(self, model, use_custom_kv_cache=False, use_custom_sdpa=False, disable_dynamic_shapes=False):
        super().__init__()
        self.model = model
        self.config = model.config
        self.use_custom_kv_cache = use_custom_kv_cache
        self.use_custom_sdpa = use_custom_sdpa
        self.disable_dynamic_shapes = disable_dynamic_shapes
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

        if not self.disable_dynamic_shapes and not is_using_hybrid_cache_wo_custom_sdpa_kv_cache:
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

    def _register_custom_attention(self, exportable_module: torch.nn.Module):
        from transformers.integrations.executorch import sdpa_mask_without_vmap
        from transformers.masking_utils import AttentionMaskInterface
        from transformers.modeling_utils import AttentionInterface

        if self.use_custom_sdpa:
            if self.use_custom_kv_cache:
                _custom_sdpa_for_ring_kv_cache = get_custom_sdpa_for_ring_kv_cache(exportable_module)
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

        from transformers.integrations.executorch import (
            TorchExportableModuleForDecoderOnlyLM,
        )

        exportable_module = TorchExportableModuleForDecoderOnlyLM(
            self.model,
            max_batch_size=1,
            max_cache_len=self.metadata.get("get_max_seq_len"),
        )
        self._register_custom_attention(exportable_module)

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
            "encoder": self.exported_encoder,  # Not called "text_encoder" because the encoder could be non-text too, e.g. Whisper.
            "text_decoder": self.exported_decoder,
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
