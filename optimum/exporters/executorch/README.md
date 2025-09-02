# Exporting Transformers Models to ExecuTorch

This directory contains everything needed to export a Transformers model to ExecuTorch.
The design philsophy is to have as little model-specific code as possible, such that any model is thereotically able to be exported if it fits a certain class.
For example, any Large Language Model thereotically should be able to be exported with `CausalLMExportableModule` in integrations.py.

## "Enabling" model enabled on Optimum?

### Are all models enabled on Optimum?
No, even though this is the goal, not every model is enabled on Optimum, whether a model is enabled depends majorly on the following:
- How much model-specific code is in Transformers for this model? e.g. does it use StaticCache, which we account for in Optimum code, or use some completely new custom cache calss?
- Do we already have the model type supported in Optimum?
- Is the model itself torch.exportable?

If the model passes the above conditions, then theoretically Optimum should be able to export it into a `.pte`, which can be used in conjunction with ExecuTorch runners to produce an E2E generation flow.
For each of the models that Optimum claims to support in the top-level README, there is a test file for it in `tests/models/test_modeling_<MODEL_NAME>` that has been used to validate correct output generation and maintain correctness in CI.
Theoretically, there are many more models that Optimum can support out of the box today, but they just haven't been validated or put into CI with a `tests/models/text_modeling_<MODELING_NAME>` file - feel free to do this a contribute to the community yourself if there is a model you are interested in that does not yet have a test file!

### How to enable a model on Optimum
If the model does not pass the above three conditions, we need to do some additional work on top of simply writing a test file to validate.

### Model-specific code is in Transformers
For this, we will need to upstream changes to the Transformers library, or update our code to match.
For instance, if hypothetically Transformers introduced a new type of cache, and this cache is used in the newest Llama model, we would need to handle this new cache type in Optimum.
Or, hypothetically if we are expecting a certain attribute in a Transformers model and it instead exists in Transformers with a slighly different name, this may be an opportunity to upstream some changes to Transformers.

#### Model type is not supported in Optimum
All the supported model types are in integrations.py:
- `CausalLMExportableModule` - LLMs (Large Language Models)
- `MultiModalTextToTextExportableModule` - Multimodal LLMs (Large Language Models with support for audio/image input)
- `MultiModalTextToTextExportableModule` - Vision Encoder backbones (such as DiT or MobileViT)
- `MaskedLMExportableModule` - Masked language models (for predicting masked characters)
- `Seq2SeqLMExportableModule` - General Seq2Seq encoder-decoder models (such as T5 and Whisper)

If the model type is not supported then we will need to write a new class for it.

## Exporting and running different model types
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
