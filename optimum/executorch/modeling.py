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

"""ExecuTorchModelForXXX classes, allowing to run ExecuTorch Models with ExecuTorch Runtime using the same API as Transformers."""

import logging
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List, Optional, Union

import torch
from huggingface_hub import hf_hub_download
from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE
from transformers import (
    AutoModelForCausalLM,
    PretrainedConfig,
    PreTrainedTokenizer,
    add_start_docstrings,
)
from transformers.utils import is_offline_mode

from executorch.extension.pybindings.portable_lib import ExecuTorchModule, _load_for_executorch

from ..exporters import TasksManager
from ..exporters.executorch import main_export
from ..modeling_base import FROM_PRETRAINED_START_DOCSTRING, OptimizedModel
from ..utils.file_utils import find_files_matching_pattern


_FILE_PATTERN = r".*\.pte$"


logger = logging.getLogger(__name__)


class ExecuTorchModelForCausalLM(OptimizedModel):
    """
    ExecuTorch model with a causal language modeling head for inference using the ExecuTorch Runtime.

    This class provides an interface for loading, running, and generating outputs from a causal language model
    optimized for ExecuTorch Runtime. It includes utilities for exporting and loading pre-trained models
    compatible with ExecuTorch runtime.

    Attributes:
        auto_model_class (`Type`):
            Associated Transformers class, `AutoModelForCausalLM`.
        model (`ExecuTorchModule`):
            The loaded ExecuTorch model.
        use_kv_cache (`bool`):
            Whether key-value caching is enabled. For performance reasons, the exported model is
            optimized to use a static cache.
        max_cache_size (`int`):
            Maximum sequence length supported by the cache.
        max_batch_size (`int`):
            Maximum supported batch size.
        dtype (`str`):
            Data type of the model parameters.
        bos_token_id (`int`):
            Beginning-of-sequence token ID.
        eos_token_id (`int`):
            End-of-sequence token ID.
        vocab_size (`int`):
            Size of the model vocabulary.
    """

    auto_model_class = AutoModelForCausalLM

    def __init__(self, model: "ExecuTorchModule", config: "PretrainedConfig"):
        super().__init__(model, config)
        metadata = self.model.method_names()
        logging.info(f"Load all static methods: {metadata}")
        if "use_kv_cache" in metadata:
            self.use_kv_cache = self.model.run_method("use_kv_cache")[0]
        if "get_max_seq_len" in metadata:
            self.max_cache_size = self.model.run_method("get_max_seq_len")[0]
        if "get_max_batch_size" in metadata:
            self.max_batch_size = self.model.run_method("get_max_batch_size")[0]
        if "get_dtype" in metadata:
            self.dtype = self.model.run_method("get_dtype")[0]
        if "get_bos_id" in metadata:
            self.bos_token_id = self.model.run_method("get_bos_id")[0]
        if "get_eos_id" in metadata:
            self.eos_token_id = self.model.run_method("get_eos_id")[0]
        if "get_vocab_size" in metadata:
            self.vocab_size = self.model.run_method("get_vocab_size")[0]

    def forward(
        self,
        input_ids: torch.Tensor,
        cache_position: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass of the model, which is compatible with the ExecuTorch runtime for LLM.

        Args:
            input_ids (`torch.Tensor`): Tensor representing current input token id to the model.
            cache_position (`torch.Tensor`): Tensor representing current input position in the cache.

        Returns:
            torch.Tensor: Logits output from the model.
        """
        return self.model.forward((input_ids, cache_position))[0]

    @classmethod
    def _from_pretrained(
        cls,
        model_id: Union[str, Path],
        config: PretrainedConfig,
        token: Optional[Union[bool, str]] = None,
        revision: Optional[str] = None,
        subfolder: str = "",
        force_download: bool = False,
        local_files_only: bool = False,
        cache_dir: str = HUGGINGFACE_HUB_CACHE,
        file_name: Optional[str] = None,
        **kwargs,
    ) -> "ExecuTorchModelForCausalLM":
        model_path = Path(model_id)
        default_file_name = file_name or "model.pte"

        if local_files_only:
            object_id = str(model_id).replace(os.sep, "--")
            cached_model_dir = os.path.join(cache_dir, f"models--{object_id}")
            refs_file = os.path.join(os.path.join(cached_model_dir, "refs"), revision or "main")
            with open(refs_file) as f:
                _revision = f.read()
            model_dir = os.path.join(cached_model_dir, "snapshots", _revision)
        else:
            model_dir = str(model_id)

        pte_files = find_files_matching_pattern(
            model_dir,
            _FILE_PATTERN,
            glob_pattern="**/*.pte",
            subfolder=subfolder,
            token=token,
            revision=revision,
        )

        model_path = Path(model_dir)
        if len(pte_files) == 0:
            raise FileNotFoundError(f"Could not find any ExecuTorch model file in {model_dir}")
        if len(pte_files) == 1 and file_name and file_name != pte_files[0].name:
            raise FileNotFoundError(f"Trying to load {file_name} but only found {pte_files[0].name}")

        file_name = pte_files[0].name
        subfolder = pte_files[0].parent

        if len(pte_files) > 1:
            for file in pte_files:
                if file.name == default_file_name:
                    file_name = file.name
                    subfolder = file.parent
                    break

            logger.warning(
                f"Too many ExecuTorch model files were found in {' ,'.join(map(str, pte_files))}. "
                "specify which one to load by using the `file_name` and/or the `subfolder` arguments. "
                f"Loading the file {file_name} in the subfolder {subfolder}."
            )

        if model_path.is_dir():
            model_path = subfolder
            subfolder = ""

        model_cache_path = cls._cached_file(
            model_path=model_path,
            token=token,
            revision=revision,
            force_download=force_download,
            cache_dir=cache_dir,
            file_name=default_file_name,
            subfolder=subfolder,
            local_files_only=local_files_only,
        )
        model = _load_for_executorch(model_cache_path)
        logging.info(f"Loaded model from {model_cache_path}")

        return cls(model, config=config)

    @staticmethod
    def _cached_file(
        model_path: Union[Path, str],
        token: Optional[Union[bool, str]] = None,
        revision: Optional[str] = None,
        force_download: bool = False,
        cache_dir: Optional[str] = None,
        file_name: Optional[str] = None,
        subfolder: str = "",
        local_files_only: bool = False,
    ):
        model_path = Path(model_path)
        # locates a file in a local folder and repo, downloads and cache it if necessary.
        if model_path.is_dir():
            model_cache_path = os.path.join(model_path, subfolder, file_name)
        else:
            model_cache_path = hf_hub_download(
                repo_id=model_path.as_posix(),
                filename=file_name,
                subfolder=subfolder,
                token=token,
                revision=revision,
                cache_dir=cache_dir,
                force_download=force_download,
                local_files_only=local_files_only,
            )

        return model_cache_path

    @classmethod
    def _export(
        cls,
        model_id: str,
        recipe: str,
        config: PretrainedConfig,
        token: Optional[Union[bool, str]] = None,
        revision: Optional[str] = None,
        cache_dir: str = HUGGINGFACE_HUB_CACHE,
        trust_remote_code: bool = False,
        subfolder: str = "",
        force_download: bool = False,
        local_files_only: bool = False,
        **kwargs,
    ):
        task = kwargs.pop("task", None)
        if task is not None:
            logger.warning(f"task was provided and set to {task} but not used, will be ignored")

        save_dir = TemporaryDirectory()
        save_dir_path = Path(save_dir.name)

        # Export to ExecuTorch and save the pte file to the temporary directory
        main_export(
            model_name_or_path=model_id,
            output_dir=save_dir_path,
            task=TasksManager.infer_task_from_model(cls.auto_model_class),
            recipe=recipe,
            subfolder=subfolder,
            revision=revision,
            cache_dir=cache_dir,
            token=token,
            local_files_only=local_files_only,
            force_download=force_download,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )
        return cls._from_pretrained(model_id=save_dir_path, config=config)

    def _save_pretrained(self, save_directory):
        """
        Saves a model weights into a directory, so that it can be re-loaded using the
        [`from_pretrained`] class method.
        """
        raise NotImplementedError

    def generate(
        self,
        prompt_tokens: List[int],
        echo: bool = False,
        pos_base: int = 0,
        max_seq_len: Optional[int] = None,
    ) -> List[int]:
        """
        Generate tokens from a prompt using the ExecuTorch model.

        Args:
            prompt_tokens (List[int]):
                List of token IDs representing the prompt.
            echo (`bool`, *optional*):
                Whether to include prompt tokens in the generated output. Defaults to `False`.
            pos_base (`int`, *optional*):
                Base position for the prompt tokens. Defaults to 0.
            max_seq_len (`int`, *optional*):
                Maximum sequence length for the generated output.
                Defaults to None and uses the model's `max_cache_size` attribute.
                Will be truncated to maximal cache size if larger than `max_cache_size`.

        Returns:
            List[int]: List of generated token IDs.

        Note:
            Temporarily implemented this method in Python due to limited access to ExecuTorch's c++ LLM runner via pybind.
            Expect improvements to the pybind interface in ExecuTorch version 0.4.1.
        """
        self.device = torch.device("cpu")
        if max_seq_len is None:
            # Default to max_cache_size if max_seq_len is not specified
            max_seq_len = self.max_cache_size
        elif max_seq_len > self.max_cache_size:
            logging.warning(
                f"max_seq_len={max_seq_len} is larger than max_cache_size={self.max_cache_size}. Generating tokens will be truncated to max_cache_size."
            )
            max_seq_len = self.max_cache_size
        generated_tokens = []

        # prefill
        for i, prompt_token in enumerate(prompt_tokens):
            logits = self.forward(
                input_ids=torch.tensor([prompt_token], dtype=torch.long, device=self.device).unsqueeze(0),
                cache_position=torch.tensor([i], dtype=torch.long, device=self.device),
            )

        next_token = torch.argmax(logits, dim=-1).item()
        generated_tokens = prompt_tokens + [next_token]

        while len(generated_tokens) < max_seq_len:
            logits = self.forward(
                input_ids=torch.tensor([next_token], dtype=torch.long, device=self.device).unsqueeze(0),
                cache_position=torch.tensor(
                    [pos_base + len(generated_tokens) - 1],
                    dtype=torch.long,
                    device=self.device,
                ),
            )
            next_token = torch.argmax(logits, dim=-1).item()
            generated_tokens.append(next_token)
            if next_token == self.eos_token_id:
                break

        return generated_tokens if echo else generated_tokens[len(prompt_tokens) :]

    def text_generation(
        self,
        tokenizer: "PreTrainedTokenizer",
        prompt: str,
        echo: bool = True,
        max_seq_len: Optional[int] = None,
    ):
        """
        Perform text generation task for a given prompt using the ExecuTorch model.

        Args:
            tokenizer (`PreTrainedTokenizer`):
                The tokenizer used to encode and decode the prompt and output.
            prompt (`str`):
                The text prompt to complete.
            echo (`bool`, *optional*):
                Whether to include prompt tokens in the generated output. Defaults to `True`.
            max_seq_len (`int`, *optional*):
                Maximum sequence length for the generated output.
                Defaults to None and uses the model's `max_cache_size` attribute.
                Will be truncated to maximal cache size if larger than `max_cache_size`.
        """
        self.tokenizer = tokenizer

        # Sanity check
        if self.tokenizer.bos_token_id is not None and self.tokenizer.bos_token_id != self.bos_token_id:
            raise ValueError(
                f"The tokenizer's bos_token_id={self.tokenizer.bos_token_id} must be the same as the model's bos_token_id={self.bos_token_id}."
            )
        if self.tokenizer.eos_token_id is not None and self.tokenizer.eos_token_id != self.eos_token_id:
            raise ValueError(
                f"The tokenizer's eos_token_id={self.tokenizer.eos_token_id} must be the same as the model's eos_token_id={self.eos_token_id}."
            )

        prompt_tokens = self.tokenizer.encode(prompt)
        generated_tokens = self.generate(
            prompt_tokens=prompt_tokens,
            echo=echo,
            max_seq_len=max_seq_len,
        )
        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

    @classmethod
    @add_start_docstrings(FROM_PRETRAINED_START_DOCSTRING)
    def from_pretrained(
        cls,
        model_id: Union[str, Path],
        export: bool = False,
        token: Optional[Union[bool, str]] = None,
        revision: Optional[str] = None,
        subfolder: str = "",
        trust_remote_code: bool = False,
        force_download: bool = False,
        local_files_only: bool = False,
        cache_dir: str = HUGGINGFACE_HUB_CACHE,
        config: Optional[PretrainedConfig] = None,
        **kwargs,
    ):
        if isinstance(model_id, Path):
            model_id = model_id.as_posix()

        if is_offline_mode() and not local_files_only:
            logger.info("Offline mode: setting `local_files_only=True`")
            local_files_only = True

        _export = export
        try:
            if local_files_only:
                object_id = model_id.replace(os.sep, "--")
                cached_model_dir = os.path.join(cache_dir, f"models--{object_id}")
                refs_file = os.path.join(os.path.join(cached_model_dir, "refs"), revision or "main")
                with open(refs_file) as f:
                    _revision = f.read()
                model_dir = os.path.join(cached_model_dir, "snapshots", _revision)
            else:
                model_dir = model_id

            pte_files = find_files_matching_pattern(
                model_dir,
                pattern=_FILE_PATTERN,
                glob_pattern="**/*.pte",
                subfolder=subfolder,
                token=token,
                revision=revision,
            )

            _export = len(pte_files) == 0
            if _export ^ export:
                if export:
                    logger.warning(
                        f"The model {model_id} was already converted to the ExecuTorch IR but got `export=True`, the model will be converted to ExecuTorch once again. "
                        # "Don't forget to save the resulting model with `.save_pretrained()`"
                    )
                    _export = True
                else:
                    logger.warning(
                        f"No ExecuTorch files were found for {model_id}, setting `export=True` to convert the model to the ExecuTorch IR. "
                        # "Don't forget to save the resulting model with `.save_pretrained()`"
                    )
        except Exception as exception:
            logger.warning(
                f"Could not infer whether the model was already converted or not to the ExecuTorch IR, keeping `export={export}`.\n{exception}"
            )

        from_pretrained_method = cls._export if _export else cls._from_pretrained

        return from_pretrained_method(
            model_id=model_id,
            config=config,
            revision=revision,
            cache_dir=cache_dir,
            force_download=force_download,
            token=token,
            subfolder=subfolder,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )
