# Exporting Transformers Models to ExecuTorch

This directory contains everything needed to export a Transformers model to ExecuTorch.
Optimum ExecuTorch's design philsophy is to have as little model-specific code as possible, which means all optimizations, export code, etc. are model-agnostic.
This allows us to theoretically export any model, with a few caveats which will be explained later.
For example, any Large Language Model thereotically should be able to be exported with `CausalLMExportableModule` in integrations.py.

## 💡 How to "enable" a model on Optimum
Optimum Executorch's goal is to be able to export any model through the `optimum-cli`.
Currently, the following models specified on the [homepage README](../../../README.md?tab=readme-ov-file#-supported-models) lists all of the "supported" models, but what does this mean?

👉 These supported models all have a test file associated with them, such as [Gemma3](https://github.com/huggingface/optimum-executorch/blob/main/tests/models/test_modeling_gemma3.py), which has been used to validate the E2E of the model (export + run generation loop on exported artifact).
The test file is then used in CI to guard against potential regressions.
In the aforementioned Gemma3 test file, we validate that the model is able to export and returns correct output to a various test prompt for different export configurations - now other users know that Gemma3 works and can export the model like so (just an example):
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

However, there are many models without test files in Optimum that probably still work, just no one has went through the trouble of validating it.
This is where the community comes in - feel free to contribute if there is a model you are interested in that does not yet have a test file!

If you run into any issues, they will most likely stem from the following:
- ❓ How much model-specific code is in Transformers for this model? e.g. does it use StaticCache, which we account for in Optimum code, or use some completely new custom cache class?
- ❓ Do we already have the model type supported in Optimum?
- ❓ Is the model itself torch.export-able?

### ❌ Model-specific code is in Transformers
To address this issue, we will need to upstream changes to the Transformers library, or update our code to match.
For instance, if hypothetically Transformers introduced a new type of cache, and this cache is used in the newest Llama model, we would need to handle this new cache type in Optimum.
Or, hypothetically if we are expecting a certain attribute in a Transformers model and it instead exists in Transformers with a slighly different name, this may be an opportunity to upstream some naming standardization changes to Transformers.

### ❌ Model type is not supported in Optimum
All of the supported model types are in [integrations.py](https://github.com/huggingface/optimum-executorch/blob/main/optimum/exporters/executorch/integrations.py), which contains wrapper classes that facilitate torch.export()ing a model:
- `CausalLMExportableModule` - LLMs (Large Language Models)
- `MultiModalTextToTextExportableModule` - Multimodal LLMs (Large Language Models with support for audio/image input)
- `MultiModalTextToTextExportableModule` - Vision Encoder backbones (such as DiT or MobileViT)
- `MaskedLMExportableModule` - Masked language models (for predicting masked characters)
- `Seq2SeqLMExportableModule` - General Seq2Seq encoder-decoder models (such as T5 and Whisper)

This is where most of the complexity of enabling a model on Optimum arises from, since post torch.export() every model follows the same flow per backend for transforming the torch.export() artifact into an Excecutorch `.pte` artifact.
If the model type doesn't exist in then we will need to write a new class for it.

### ❌ Model is not torch.exportable
To address this issue, we will need to upstream changes to the model's modeling file in Transformers to make the model exportable.

# Exporting and running different model types
<In progress>

### LLMs (Large Language Models)
<In progress>

### Multimodal LLMs

The export will produce a `.pte` with the following methods:
- `text_decoder`: the text decoder or language model backbone
- `audio_encoder` or `vision_encoder`: the encoder which feeds into the decoder
- `token_embedding`: the embedding layer of the language model backbone
  -  This is needed in order to cleanly separate the entire multimodal model into subgraphs. The text decoder subgraph will take in token embeddings, so multimodal input will be processed into embeddings by the encoder while text input will be processed into embeddings by this method.

### Seq2Seq
<In progress>

The export will produce a `.pte` with the following methods:
- `text_decoder`: the decoder half of the Seq2Seq model
- `encoder`: the encoder half of the Seq2Seq model. This encode can support a variety of modalities, such as text for T5 and audio for Whisper.

### Image classification
<In progress>

### ASR (Automatic speech recognition)
<In progress>
