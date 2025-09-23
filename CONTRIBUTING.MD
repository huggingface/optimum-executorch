Thank you for your interest in contributing to Optimum ExecuTorch!

## Developing Optimum ExecuTorch

### Setting up the development environment
```
python install_dev.py
```

### Testing local chagnes
Optimum ExecuTorch does not have an editable install at the moment, so to test your local changes, you will need to reinstall.
To prevent the reinstall from overwriting other dependencies, some of which you may have modified, you can run the following ahead of your test:
```
pip install --no-deps --no-build-isolation .
```

An example command for testing local changes to Gemma3:
```
pip install --no-deps --no-build-isolation .
RUN_SLOW=1 python -m pytest tests/models/test_modeling_gemma3.py -s -k test_gemma3_image_vision_with_custom_sdpa_kv_cache_8da4w_8we --log-cli-level=INFO
```

To run tests marked with `@slow`, just set `RUN_SLOW=1`.

## Enabling a new model on Optimum

Our design philsophy is to have as little model-specific code as possible, which means all optimizations, export code, etc. are model-agnostic.
This allows us to theoretically export any new model straight from the source, with a few caveats which will be explained later.
For example, most Large Language Models should be able to be exported using this library.

### 💡 How to "enable" a model on Optimum
❓ Currently, the [homepage README](README.md?tab=readme-ov-file#-supported-models) lists all of the "supported" models. What does this mean, and what about models not on this list?

👉 These supported models all have a test file associated with them, such as [Gemma3](https://github.com/huggingface/optimum-executorch/blob/main/tests/models/test_modeling_gemma3.py), which has been used to validate the E2E of the model (export + run generation loop on exported artifact).
The test file is then used in CI to guard against potential regressions.
In the Gemma3 test file, we have validated that the model is able to export and returns correct output to a test prompt for different export configurations - now other users will know that Gemma3 works and are able to export the model like so:
```
optimum-cli export executorch \
  --model google/gemma-3-1b-it \
  --task text-generation \
  --recipe xnnpack \
  --use_custom_sdpa \
  --use_custom_kv_cache \
  --qlinear 8da4w \
  --qembedding 8w
```

However, there are many models without test files in Optimum that probably still work - just that no one has went through the trouble of validating them.
This is where you come in - feel free to contribute if there is a model you are interested in that does not yet have a test file!

If you run into any issues, they will most likely stem from the following:
- ❓ How much model-specific code is in Transformers for this model? e.g. does it use StaticCache, which we account for in Optimum code, or use some completely new custom cache class?
- ❓ Do we already have the model type supported in Optimum?
- ❓ Is the model itself torch.exportable?

### ❌ Model-specific code is in Transformers
To address this issue, we will need to upstream changes to the Transformers library, or update our code to match.
For instance, if hypothetically Transformers introduced a new type of cache, and this cache is used in a new LLM, we would need to handle this new cache type in Optimum.
Or, hypothetically if we are expecting a certain attribute in a Transformers model and it exists instead with a slighly different name, this may be an opportunity to upstream some naming standardization changes to Transformers.

### ❌ Model type is not supported in Optimum
All of the supported model types are in [integrations.py](https://github.com/huggingface/optimum-executorch/blob/main/optimum/exporters/executorch/integrations.py), which contains wrapper classes that facilitate torch.exporting a model:
- `CausalLMExportableModule` - LLMs (Large Language Models)
- `MultiModalTextToTextExportableModule` - Multimodal LLMs (Large Language Models with support for audio/image input)
- `VisionEncoderExportableModule` - Vision Encoder backbones (such as DiT or MobileViT)
- `MaskedLMExportableModule` - Masked language models (for predicting masked characters)
- `Seq2SeqLMExportableModule` - General Seq2Seq encoder-decoder models (such as T5 and Whisper)

This is where most of the complexity around "enabling" a model on Optimum arises from, since post torch.export() every model follows the same flow per backend for transforming the torch.export() artifact into an Excecutorch `.pte` artifact.
If the model type doesn't exist in Optimum then we will need to write a new class for it.

### ❌ Model is not torch.exportable
To address this issue, we will need to upstream changes to the model's modeling file in Transformers to make the model exportable.
After doing this, it's a good idea to add a torch.export test to guard against future regressions (which tend to happen frequently since Transformers moves fast).
[Here](https://github.com/huggingface/transformers/blob/87f38dbfcec48027d4bf2ea7ec8b8eecd5a7bc85/tests/models/smollm3/test_modeling_smollm3.py#L175) is an example.

Once you have a PR up for adding the test to the repo, feel free to edit the homepage README to include the new model.
