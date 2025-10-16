# Exporting Transformers Models to ExecuTorch

Optimum ExecuTorch enables exporting models from Transformers to ExecuTorch.
The models supported by Optimum ExecuTorch are listed [here](../../../README.md#-supported-models).

### LLMs (Large Language Models)
LLMs can be exported using the `text-generation` task like so:
```
optimum-cli export executorch \
  --model <model-id> \
  --task text-generation \
  --recipe xnnpack \
  --use_custom_sdpa \
  --use_custom_kv_cache \
  --qlinear 8da4w \
  --qembedding 8w
  ...etc...
```

The export will produce a `.pte` with a single forward method for the decoder: `model`.

Note that most of the arguments here are only applicable to LLMs (multimodal included):
```
--use_custom_sdpa \
--use_custom_kv_cache \
--qlinear 8da4w \
--qembedding 8w
```

### Multimodal LLMs
Multimodal LLMs can be exported using the `multimodal-text-to-text` task like so:
```
optimum-cli export executorch \
  --model mistralai/Voxtral-Mini-3B-2507 \
  --task multimodal-text-to-text \
  --recipe xnnpack \
  --use_custom_sdpa \
  --use_custom_kv_cache \
  --qlinear 8da4w \
  --qembedding 8w
  ...etc...
```

The export will produce a `.pte` with the following methods:
- `text_decoder`: the text decoder or language model backbone
- `audio_encoder` or `vision_encoder`: the encoder which feeds into the decoder
- `token_embedding`: the embedding layer of the language model backbone
  -  This is needed in order to cleanly separate the entire multimodal model into subgraphs. The text decoder subgraph will take in token embeddings, so multimodal input will be processed into embeddings by the encoder while text input will be processed into embeddings by this method.

### Seq2Seq
Seq2Seq models can be exported using the `text2text-generation` task like so:
```
optimum-cli export executorch \
  --model google-t5/t5-small \
  --task text2text-generation \
  --recipe xnnpack
```

The export will produce a `.pte` with the following methods:
- `text_decoder`: the decoder half of the Seq2Seq model
- `encoder`: the encoder half of the Seq2Seq model. This encoder can support a variety of modalities, such as text for T5 and audio for Whisper.

### Image classification
Image classification models can be exported using the `image-classification` task like so:
```
optimum-cli export executorch \
  --model google/vit-base-patch16-224 \
  --task image-classification \
  --recipe xnnpack
```

The export will produce a `.pte` with a single forward method for the decoder: `model`.

### ASR (Automatic speech recognition)
ASR is a special case of Seq2Seq that uses the base Seq2Seq exportable modules. It can be exported using the `automatic-speech-recognition` task like so:
```
optimum-cli export executorch \
  --model openai/whisper-tiny \
  --task automatic-speech-recognition \
  --recipe xnnpack
```

The export will produce a `.pte` with the following methods:
- `text_decoder`: the decoder half of the Seq2Seq model
- `encoder`: the encoder half of the Seq2Seq model.
