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
from executorch import version
from packaging.version import parse
from transformers import AutoTokenizer
from transformers.testing_utils import slow

from optimum.executorch import ExecuTorchModelForSeq2SeqLM


class ExecuTorchModelIntegrationTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @slow
    @pytest.mark.run_slow
    def test_t5_export_to_executorch(self):
        model_id = "google-t5/t5-small"
        task = "text2text-generation"
        recipe = "xnnpack"
        with tempfile.TemporaryDirectory() as tempdir:
            subprocess.run(
                f"optimum-cli export executorch --model {model_id} --task {task} --recipe {recipe} --output_dir {tempdir}/executorch",
                shell=True,
                check=True,
            )
            self.assertTrue(os.path.exists(f"{tempdir}/executorch/model.pte"))

    def _helper_t5_translation(self, recipe: str):
        model_id = "google/flan-t5-small"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = ExecuTorchModelForSeq2SeqLM.from_pretrained(model_id, task="text2text-generation", recipe=recipe)

        input_text = "translate English to German: How old are you?"
        generated_text = model.text_generation(
            tokenizer=tokenizer,
            prompt=input_text,
        )
        expected_text = "Wie ich er bitten?"
        logging.info(f"\nInput text:\n\t{input_text}\nGenerated text:\n\t{generated_text}")
        self.assertEqual(generated_text, expected_text)

    @slow
    @pytest.mark.run_slow
    def test_t5_translation(self):
        self._helper_t5_translation(recipe="xnnpack")

    @slow
    @pytest.mark.run_slow
    @pytest.mark.portable
    @pytest.mark.skipif(
        parse(version.__version__) < parse("0.7.0"),
        reason="Fixed on executorch >= 0.7.0",
    )
    def test_t5_translation_portable(self):
        self._helper_t5_translation(recipe="portable")

    def _helper_t5_summarization(self, recipe: str):
        model_id = "google-t5/t5-small"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = ExecuTorchModelForSeq2SeqLM.from_pretrained(model_id, task="text2text-generation", recipe=recipe)

        article = (
            " New York (CNN)When Liana Barrientos was 23 years old, she got married in Westchester County, New York. A"
            " year later, she got married again in Westchester County, but to a different man and without divorcing"
            " her first husband.  Only 18 days after that marriage, she got hitched yet again. Then, Barrientos"
            ' declared "I do" five more times, sometimes only within two weeks of each other. In 2010, she married'
            " once more, this time in the Bronx. In an application for a marriage license, she stated it was her"
            ' "first and only" marriage. Barrientos, now 39, is facing two criminal counts of "offering a false'
            ' instrument for filing in the first degree," referring to her false statements on the 2010 marriage'
            " license application, according to court documents. Prosecutors said the marriages were part of an"
            " immigration scam. On Friday, she pleaded not guilty at State Supreme Court in the Bronx, according to"
            " her attorney, Christopher Wright, who declined to comment further. After leaving court, Barrientos was"
            " arrested and charged with theft of service and criminal trespass for allegedly sneaking into the New"
            " York subway through an emergency exit, said Detective Annette Markowski, a police spokeswoman. In total,"
            " Barrientos has been married 10 times, with nine of her marriages occurring between 1999 and 2002.  All"
            " occurred either in Westchester County, Long Island, New Jersey or the Bronx. She is believed to still be"
            " married to four men, and at one time, she was married to eight men at once, prosecutors say. Prosecutors"
            " said the immigration scam involved some of her husbands, who filed for permanent residence status"
            " shortly after the marriages.  Any divorces happened only after such filings were approved. It was"
            " unclear whether any of the men will be prosecuted. The case was referred to the Bronx District"
            " Attorney's Office by Immigration and Customs Enforcement and the Department of Homeland Security's"
            ' Investigation Division. Seven of the men are from so-called "red-flagged" countries, including Egypt,'
            " Turkey, Georgia, Pakistan and Mali. Her eighth husband, Rashid Rajput, was deported in 2006 to his"
            " native Pakistan after an investigation by the Joint Terrorism Task Force."
        )
        article = "summarize: " + article.strip()

        generated_text = model.text_generation(
            tokenizer=tokenizer,
            prompt=article,
        )
        expected_text = 'a year later, she got married again in westchester county, new york. she was married to a different man, but only 18 days after that marriage. she is facing two criminal counts of "offering a false instrument"'
        logging.info(f"\nInput text:\n\t{article}\nGenerated text:\n\t{generated_text}")
        self.assertEqual(generated_text, expected_text)

    @slow
    @pytest.mark.run_slow
    def test_t5_summarization(self):
        self._helper_t5_summarization(recipe="xnnpack")

    @slow
    @pytest.mark.run_slow
    @pytest.mark.portable
    @pytest.mark.skipif(
        parse(version.__version__) < parse("0.7.0"),
        reason="Fixed on executorch >= 0.7.0",
    )
    def test_t5_summarization_portable(self):
        self._helper_t5_summarization(recipe="portable")
