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
from transformers import PreTrainedModel, StaticCache
from transformers.generation.configuration_utils import GenerationConfig


class Seq2SeqLMEncoderExportableModule(torch.nn.Module):
    """
    A wrapper module designed to make a Seq2Seq LM encoder exportable with `torch.export`.
    This module ensures that the exported encoder model is compatible with ExecuTorch.
    """

    def __init__(self, encoder_model):
        super().__init__()
        self.encoder = encoder_model

    def forward(self, input_ids):
        return self.encoder(input_ids=input_ids).last_hidden_state


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
        self.lm_head = model.lm_head
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

        # Apply language model head
        lm_logits = self.lm_head(outputs[0])

        return lm_logits


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
        self.exported_encoder = None
        self.exported_decoder = None

    def _export_encoder(self, encoder_input_ids):
        wrapped_encoder = Seq2SeqLMEncoderExportableModule(self.encoder).to("cpu").eval()

        # Define dynamic sequence length for encoder
        seq_len_dim = torch.export.Dim("encoder_seq_length", max=self.max_hidden_seq_length)

        # Export the encoder
        with torch.no_grad():
            exported_encoder = torch.export.export(
                wrapped_encoder, (encoder_input_ids,), dynamic_shapes={"input_ids": {1: seq_len_dim}}, strict=True
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

        # Define dynamic dimension for encoder output sequence length
        encoder_seq_len_dim = torch.export.Dim("encoder_hidden_seq_length", max=self.max_hidden_seq_length)

        # Export the decoder
        with torch.no_grad():
            exported_decoder = torch.export.export(
                wrapped_decoder,
                (decoder_input_ids, encoder_hidden_states, cache_position),
                dynamic_shapes={
                    "decoder_input_ids": None,
                    "encoder_hidden_states": {1: encoder_seq_len_dim},
                    "cache_position": None,
                },
                strict=True,
            )

        return exported_decoder

    def export(self, encoder_input_ids=None, decoder_input_ids=None, encoder_hidden_states=None, cache_position=None):
        example_encoder_input_ids = (
            encoder_input_ids if encoder_input_ids is not None else torch.ones((1, 10), dtype=torch.long)
        )
        example_decoder_input_ids = (
            decoder_input_ids if decoder_input_ids is not None else torch.tensor([[0]], dtype=torch.long)
        )  # Start token
        example_cache_position = cache_position if cache_position is not None else torch.tensor([0], dtype=torch.long)
        example_encoder_hidden_states = (
            encoder_hidden_states
            if encoder_hidden_states is not None
            else torch.zeros(
                (self.generation_config.cache_config.batch_size, 10, self.config.d_model), dtype=torch.float32
            )
        )
        self.exported_encoder = self._export_encoder(example_encoder_input_ids)
        self.exported_decoder = self._export_decoder(
            example_decoder_input_ids, example_encoder_hidden_states, example_cache_position
        )

        # Return self to allow chaining
        return self

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
                    decoder_input_ids, encoder_output, torch.tensor([i], dtype=torch.long)
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
