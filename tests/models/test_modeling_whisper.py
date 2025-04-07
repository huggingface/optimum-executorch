# coding=utf-8
# Copyright 2024 The HuggingFace Team. All rights reserved.
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
import os
import subprocess
import tempfile
import unittest

import pytest
import torch

from executorch.extension.pybindings.portable_lib import ExecuTorchModule
from optimum.executorch import ExecuTorchModelForSpeechSeq2Seq
from transformers import AutoTokenizer
from transformers.testing_utils import slow


class ExecuTorchModelIntegrationTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @slow
    @pytest.mark.run_slow
    def test_whisper_export_to_executorch(self):
        model_id = "openai/whisper-tiny"
        task = "automatic-speech-recognition"
        recipe = "xnnpack"
        with tempfile.TemporaryDirectory() as tempdir:
            subprocess.run(
                f"optimum-cli export executorch --model {model_id} --task {task} --recipe {recipe} --output_dir {tempdir}/executorch",
                shell=True,
                check=True,
            )
            self.assertTrue(os.path.exists(f"{tempdir}/executorch/encoder.pte"))
            self.assertTrue(os.path.exists(f"{tempdir}/executorch/decoder.pte"))

    @slow
    @pytest.mark.run_slow
    def test_whisper_transcription(self):
        model_id = "openai/whisper-tiny"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = ExecuTorchModelForSpeechSeq2Seq.from_pretrained(model_id, recipe="xnnpack")

        self.assertIsInstance(model, ExecuTorchModelForSpeechSeq2Seq)
        self.assertTrue(hasattr(model, "encoder"))
        self.assertIsInstance(model.encoder, ExecuTorchModule)
        self.assertTrue(hasattr(model, "decoder"))
        self.assertIsInstance(model.decoder, ExecuTorchModule)

        # Set manual seed for reproducibility, Whisper could possibly hallucinate tokens
        # in some cases if this is not set.
        torch.manual_seed(11)
        input_features = torch.rand(1, 80, 3000)
        generated_transcription = model.transcribe(tokenizer, input_features)
        expected_text = ""
        logging.info(
            f"\nExpected transcription:\n\t{expected_text}\nGenerated transcription:\n\t{generated_transcription}"
        )
        self.assertEqual(generated_transcription, expected_text)
