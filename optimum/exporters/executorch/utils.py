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

import copy
import io
import logging
from typing import Any, Dict, List, Optional, Set

import torch
import transformers
from transformers import GenerationConfig, PretrainedConfig
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils import PreTrainedTokenizer


def save_config_to_constant_methods(
    config: PretrainedConfig,
    generation_config: Optional[GenerationConfig] = None,
    processor_config: Optional[dict] = None,
    **kwargs,
):
    # Initialize metadata with values from model config
    head_dim = None
    if (
        hasattr(config, "hidden_size")
        and hasattr(config, "num_attention_heads")
        and isinstance(config.num_attention_heads, int)
    ):
        head_dim = config.hidden_size / config.num_attention_heads
    metadata = {
        "get_dtype": 5 if config.torch_dtype == torch.float16 else 6,
        "get_bos_id": getattr(config, "bos_token_id", None),
        "get_eos_id": getattr(config, "eos_token_id", None),
        "get_head_dim": head_dim,
        "get_n_kv_heads": getattr(config, "num_key_value_heads", None),
        "get_n_layers": getattr(config, "num_hidden_layers", None),
        "get_vocab_size": getattr(config, "vocab_size", None),
        "get_max_batch_size": 1,
        "get_max_seq_len": getattr(config, "max_position_embeddings", None),
        "use_kv_cache": getattr(generation_config, "use_cache", None),
        "sliding_window": getattr(config, "sliding_window", None),
        "decoder_start_token_id": getattr(config, "decoder_start_token_id", None),
        "use_sdpa_with_kv_cache": "custom_sdpa" in config._attn_implementation,
        "enable_dynamic_shape": kwargs.get("enable_dynamic_shape", True),
    }

    # Include processor_config keys in metadata if provided
    if processor_config is not None:
        metadata.update(processor_config)

    # Combine/override with any additional kwargs and filter out None values
    combined_metadata = {k: v for k, v in {**metadata, **kwargs}.items() if v is not None}
    return combined_metadata


def apply_chat_template_with_fallback(processor, conversation, **kwargs):
    """
    Apply chat template with fallback for external processors.

    For duck-typed external processors that aren't defined in Transformers, e.g.
    Voxtral's processor which is defined in mistral-common.
    These processors aren't guaranteed to have some of the other kwargs such as
    "add_generation_prompt".

    Args:
        processor: The processor instance
        conversation: The conversation to process
        **kwargs: Additional keyword arguments to pass to apply_chat_template

    Returns:
        The processed inputs from apply_chat_template
    """
    try:
        return processor.apply_chat_template(conversation, **kwargs)
    except ValueError:
        # Fallback for external processors - just pass the conversation
        return processor.apply_chat_template(conversation)


def verify_eos_tokens_in_pretrained_tokenizer(model_eos_ids: List[int], tokenizer: PreTrainedTokenizer) -> bool:
    """
    Verifies that the model's EOS token IDs are present in the tokenizer's
    set of potential end-of-sequence tokens.

    Args:
        model_eos_ids: A list of EOS token IDs recorded int the PTE file (the source of truth).
        tokenizer: The Hugging Face tokenizer instance to check.

    Returns:
        True if at least one model EOS ID is found among the tokenizer's potential
        EOS tokens, False otherwise.
    """
    if not model_eos_ids:
        print("Warning: model_eos_ids list is empty. No verification can be performed.")
        return True

    candidate_eos_ids: Set[int] = set()

    # 1. Check primary eos_token and pad_token attributes
    if tokenizer.eos_token_id is not None:
        candidate_eos_ids.add(tokenizer.eos_token_id)
    if tokenizer.pad_token_id is not None:
        candidate_eos_ids.add(tokenizer.pad_token_id)

    # 2. Check all tokens listed in the special_tokens_map
    for token_string in tokenizer.special_tokens_map.values():
        if token_string:
            # Use convert_tokens_to_ids for robustness
            token_id = tokenizer.convert_tokens_to_ids(token_string)
            if isinstance(token_id, int):
                candidate_eos_ids.add(token_id)

    # 3. Check added tokens for "end-of-X" patterns
    for token_id, added_token in tokenizer.added_tokens_decoder.items():
        token_str = added_token.content.lower()
        # Heuristic to find tokens that signify an end
        if "end" in token_str or token_str.startswith("</"):
            candidate_eos_ids.add(token_id)

    # The check: is any "true" ID present in the candidate set?
    is_valid = any(model_id in candidate_eos_ids for model_id in model_eos_ids)

    return is_valid


def process_conversation_inputs(
    processor: ProcessorMixin,
    tokenizer: PreTrainedTokenizer,
    input_conversation: List[Dict[str, Any]],
):
    """
    Process input conversation for multimodal models.

    This function handles the preprocessing of conversation inputs, with special handling for
    GraniteSpeechProcessor which requires extracting and processing audio content from conversations
    prior to feeding into the processor.

    Args:
        processor: The processor to use for input processing
        tokenizer: The tokenizer to use for text processing
        input_conversation: List of conversation messages, may contain audio content

    Returns:
        Processed inputs ready for model consumption
    """
    if isinstance(processor, transformers.models.granite_speech.processing_granite_speech.GraniteSpeechProcessor):
        import requests
        import torchaudio

        conversation = copy.deepcopy(input_conversation)
        audio_path = None

        # Extract audio content and remove from conversation
        audio_items = [(i, item) for i, item in enumerate(conversation) if item.get("type") == "audio"]
        if audio_items:
            idx, audio_item = audio_items[0]
            audio_path = audio_item["content"]
            # Remove the audio content from the input conversation since it
            # is handled outside for Granite.
            del conversation[idx]
        else:
            raise ValueError("No audio content found in conversation")

        # Download and process audio
        try:
            resp = requests.get(audio_path)
            resp.raise_for_status()
            buf = io.BytesIO(resp.content)
        except requests.exceptions.RequestException:
            print("Could not download input audio file.")

        wav, sampling_rate = torchaudio.load(buf, normalize=True)
        if wav.shape[0] != 1:
            wav = wav.mean(dim=0, keepdim=True)  # Convert stereo to mono.
            logging.warning("Resampled audio stereo to mono")
        if sampling_rate != 16000:
            wav = torchaudio.functional.resample(wav, sampling_rate, 16000)
            logging.warning(f"Resampled audio from {sampling_rate}Hz to 16000Hz")

        # Generate text prompt and process with audio
        prompt = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        inputs = processor(prompt, wav, return_tensors="pt")
    else:
        # Standard processing for other processors
        inputs = apply_chat_template_with_fallback(
            processor,
            input_conversation,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

    return inputs
