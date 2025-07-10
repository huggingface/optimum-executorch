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
from datasets import load_dataset
from executorch.extension.pybindings.portable_lib import ExecuTorchModule
from transformers import AutoProcessor, AutoTokenizer
from transformers.testing_utils import slow

from optimum.executorch import ExecuTorchModelForSpeechSeq2Seq
from optimum.utils.import_utils import is_transformers_version


os.environ["TOKENIZERS_PARALLELISM"] = "false"


@pytest.mark.skipif(
    is_transformers_version(">", "4.52.4"),
    reason="Need to fix in the transformers due to attention refactor https://github.com/huggingface/transformers/pull/38235",
)
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

    def _helper_whisper_transcription(self, recipe: str):
        model_id = "openai/whisper-tiny"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = ExecuTorchModelForSpeechSeq2Seq.from_pretrained(model_id, recipe=recipe)
        processor = AutoProcessor.from_pretrained(model_id)

        self.assertIsInstance(model, ExecuTorchModelForSpeechSeq2Seq)
        self.assertTrue(hasattr(model, "encoder"))
        self.assertIsInstance(model.encoder, ExecuTorchModule)
        self.assertTrue(hasattr(model, "decoder"))
        self.assertIsInstance(model.decoder, ExecuTorchModule)

        dataset = load_dataset("distil-whisper/librispeech_long", "clean", split="validation")
        sample = dataset[0]["audio"]

        input_features = processor(
            sample["array"], return_tensors="pt", truncation=False, sampling_rate=sample["sampling_rate"]
        ).input_features
        # Current implementation of the transcibe method accepts up to 30 seconds of audio, therefore I trim the audio here.
        input_features_trimmed = input_features[:, :, :3000].contiguous()

        generated_transcription = model.transcribe(tokenizer, input_features_trimmed)
        expected_text = " Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel. Nor is Mr. Quilter's manner less interesting than his matter. He tells us that at this festive season of the year, with Christmas and roast beef looming before us, similarly drawn from eating and its results occur most readily to the mind. He has grave doubts whether Sir Frederick Latins work is really Greek after all, and can discover that."
        logging.info(
            f"\nExpected transcription:\n\t{expected_text}\nGenerated transcription:\n\t{generated_transcription}"
        )
        self.assertEqual(generated_transcription, expected_text)

    @slow
    @pytest.mark.run_slow
    def test_whisper_transcription(self):
        self._helper_whisper_transcription(recipe="xnnpack")

    @slow
    @pytest.mark.run_slow
    @pytest.mark.portable
    def test_whisper_transcription_portable(self):
        self._helper_whisper_transcription(recipe="portable")
