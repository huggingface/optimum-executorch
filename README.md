<div align="center">

<img src="https://huggingface.co/datasets/optimum/documentation-images/resolve/main/executorch/logo/optimum-executorch.png" width=80%>

# 🤗 Optimum ExecuTorch

**Optimize and deploy Hugging Face models with ExecuTorch**

[Documentation](https://huggingface.co/docs/optimum-executorch/en/index) | [ExecuTorch](https://github.com/pytorch/executorch) | [Hugging Face](https://huggingface.co/)

</div>

## 📋 Overview

Optimum ExecuTorch enables efficient deployment of transformer models using Meta's ExecuTorch framework. It provides:
- 🔄 Easy conversion of Hugging Face models to ExecuTorch format
- ⚡ Optimized inference with hardware-specific optimizations
- 🤝 Seamless integration with Hugging Face Transformers
- 📱 Efficient deployment on various devices

## ⚡ Quick Installation

### 1. Create a virtual environment
Install [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) on your machine. Then, create a virtual environment to manage our dependencies.
```
conda create -n optimum-executorch python=3.11
conda activate optimum-executorch
```

### 2. Install optimum-executorch from source
```
git clone https://github.com/huggingface/optimum-executorch.git
cd optimum-executorch
pip install '.[dev]'
```

- 🔜 Install from pypi coming soon...

### 3. Install dependencies in dev mode

To access every available optimization and experiment with the newest features, run:
```
python install_dev.py
```

This script will install `executorch`, `torch`, `torchao`, `transformers`, etc. from nightly builds or from source to access the latest models and optimizations.

To leave an existing ExecuTorch installation untouched, run `install_dev.py` with `--skip_override_torch` to prevent it from being overwritten.

## 🎯 Quick Start

There are two ways to use Optimum ExecuTorch:

### Option 1: Export and Load in One Python API
```python
from optimum.executorch import ExecuTorchModelForCausalLM
from transformers import AutoTokenizer

# Load and export the model on-the-fly
model_id = "HuggingFaceTB/SmolLM2-135M-Instruct"
model = ExecuTorchModelForCausalLM.from_pretrained(
    model_id,
    recipe="xnnpack",
    attn_implementation="custom_sdpa",  # Use custom SDPA implementation for better performance
    use_custom_kv_cache=True,  # Use custom KV cache for better performance
    **{"qlinear": "8da4w", "qembedding": "8w"},  # Quantize linear and embedding layers
)

# Generate text right away
tokenizer = AutoTokenizer.from_pretrained(model_id)
generated_text = model.text_generation(
    tokenizer=tokenizer,
    prompt="Once upon a time",
    max_seq_len=128,
)
print(generated_text)
```

> **Note:** If an ExecuTorch model is already cached on the Hugging Face Hub, the API will automatically skip the export step and load the cached `.pte` file. To test this, replace the `model_id` in the example above with `"executorch-community/SmolLM2-135M"`, where the `.pte` file is pre-cached. Additionally, the `.pte` file can be directly associated with the eager model, as demonstrated in this [example](https://huggingface.co/optimum-internal-testing/tiny-random-llama/tree/executorch).


### Option 2: Export and Load Separately

#### Step 1: Export your model
Use the CLI tool to convert your model to ExecuTorch format:
```
optimum-cli export executorch \
    --model "HuggingFaceTB/SmolLM2-135M-Instruct" \
    --task "text-generation" \
    --recipe "xnnpack" \
    --use_custom_sdpa \
    --use_custom_kv_cache \
    --qlinear 8da4w \
    --qembedding 8w \
    --output_dir="hf_smollm2"
```
Explore the various export options by running the command: `optimum-cli export executorch --help`.
To read more about how to export different types of models on Optimum ExecuTorch, please revert to the export [README](optimum/exporters/executorch/README.md).

#### Step 2: Validate the Exported Model on Host Using the Python API
Use the exported model for text generation:
```python
from optimum.executorch import ExecuTorchModelForCausalLM
from transformers import AutoTokenizer

# Load the exported model
model = ExecuTorchModelForCausalLM.from_pretrained("./hf_smollm2")

# Initialize tokenizer and generate text
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
generated_text = model.text_generation(
    tokenizer=tokenizer,
    prompt="Once upon a time",
    max_seq_len=128
)
print(generated_text)
```

#### Step 3: Run inference on-device
To perform on-device inference, you can use ExecuTorch’s sample runner or the example iOS/Android applications. For detailed instructions, refer to the [ExecuTorch Sample Runner guide](https://github.com/pytorch/executorch/blob/main/examples/models/qwen3/README.md#example-run).

## ⚙️ Optimizations

### Custom Operators
Optimum transformer models utilize:
- A [**custom SDPA**](https://github.com/pytorch/executorch/blob/a4322c71c3a97e79e0454a8223db214b010f1193/extension/llm/README.md?plain=1#L40) for CPU based on Flash Attention, boosting performance by around **3x** compared to default SDPA.
- A **custom KV cache** that uses a custom op for efficient in-place cache update on CPU, boosting performance by **2.5x** compared to default static KV cache.

### Backends Delegation
Currently, **Optimum-ExecuTorch** supports the [XNNPACK Backend](https://pytorch.org/executorch/main/backends-xnnpack.html) for CPU and [CoreML Backend](https://docs.pytorch.org/executorch/stable/backends-coreml.html) for GPU on Apple devices.

For a comprehensive overview of all backends supported by ExecuTorch, please refer to the [ExecuTorch Backend Overview](https://pytorch.org/executorch/main/backends-overview.html).

### Quantization
We currently support Post-Training Quantization (PTQ) for linear layers and embeddings using the [TorchAO](https://github.com/pytorch/ao) quantization library.


## 🤗 Supported Models

The following models have been successfully tested with Executorch. For details on the specific optimizations supported and how to use them for each model, please consult their respective test files in the [`tests/models/`](https://github.com/huggingface/optimum-executorch/tree/main/tests/models) directory.

### Text Models
We currently support a wide range of popular transformer models, including encoder-only, decoder-only, and encoder-decoder architectures, as well as models specialized for various tasks like text generation, translation, summarization, and mask prediction, etc. These models reflect the current trends and popularity across the Hugging Face community:
#### LLMs (Large Language Models)
##### Decoder-only
- [Codegen](https://huggingface.co/Salesforce/codegen-350M-mono): Salesforce's `codegen-350M-mono` and its variants
- [Gemma](https://huggingface.co/google/gemma-2b): `Gemma-2b` and its variants
- [Gemma2](https://huggingface.co/google/gemma-2-2b): `Gemma-2-2b` and its variants
- [Gemma3](https://huggingface.co/google/gemma-3-1b-it): `Gemma-3-1b` and its variants (💡[**NEW**] 270M, 1B)
- [Glm](https://huggingface.co/THUDM/glm-edge-1.5b-chat): `glm-edge-1.5b` and its variants
- [Gpt2](https://huggingface.co/AI-Sweden-Models/gpt-sw3-126m): `gpt-sw3-126m` and its variants
- [GptJ](https://huggingface.co/Milos/slovak-gpt-j-405M): `gpt-j-405M` and its variants
- [GptNeoX](https://huggingface.co/EleutherAI/pythia-14m): EleutherAI's `pythia-14m` and its variants
- [GptNeoXJapanese](https://huggingface.co/abeja/gpt-neox-japanese-2.7b): `gpt-neox-japanese-2.7b` and its variants
- [Granite](https://huggingface.co/ibm-granite/granite-3.3-2b-instruct): `granite-3.3-2b-instruct` and its variants
- [Llama](https://huggingface.co/meta-llama/Llama-3.2-1B): `Llama-3.2-1B` and its variants
- [Mistral](https://huggingface.co/ministral/Ministral-3b-instruct): `Ministral-3b-instruct` and its variants
- [Qwen2](https://huggingface.co/Qwen/Qwen2.5-0.5B): `Qwen2.5-0.5B` and its variants
- [Qwen3](https://huggingface.co/Qwen/Qwen3-0.6B): `Qwen3-0.6B`, `Qwen3-Embedding-0.6B` and other variants
- [Olmo](https://huggingface.co/allenai/OLMo-1B-hf): `OLMo-1B-hf` and its variants
- [Phi](https://huggingface.co/johnsnowlabs/JSL-MedPhi2-2.7B): `JSL-MedPhi2-2.7B` and its variants
- [Phi4](https://huggingface.co/microsoft/Phi-4-mini-instruct): `Phi-4-mini-instruct` and its variants
- [Smollm](https://huggingface.co/HuggingFaceTB/SmolLM2-135M): 🤗 `SmolLM2-135M` and its variants
- [Smollm3](https://huggingface.co/HuggingFaceTB/SmolLM3-3B): 🤗 `SmolLM3-3B` and its variants
- [Starcoder2](https://huggingface.co/bigcode/starcoder2-3b): `starcoder2-3b` and its variants
##### Encoder-decoder (Seq2Seq)
- [T5](https://huggingface.co/google-t5/t5-small): Google's `T5` and its variants
#### NLU (Natural Language Understanding)
- [Albert](https://huggingface.co/albert/albert-base-v2): `albert-base-v2` and its variants
- [Bert](https://huggingface.co/google-bert/bert-base-uncased): Google's `bert-base-uncased` and its variants
- [Distilbert](https://huggingface.co/distilbert/distilbert-base-uncased): `distilbert-base-uncased` and its variants
- [Eurobert](https://huggingface.co/EuroBERT/EuroBERT-210m): `EuroBERT-210m` and its variants
- [Roberta](https://huggingface.co/FacebookAI/xlm-roberta-base): FacebookAI's `xlm-roberta-base` and its variants

### Vision Models
- [Cvt](https://huggingface.co/microsoft/cvt-13): Convolutional Vision Transformer
- [Deit](https://huggingface.co/facebook/deit-base-distilled-patch16-224): Distilled Data-efficient Image Transformer (base-sized)
- [Dit](https://huggingface.co/microsoft/dit-base-finetuned-rvlcdip): Document Image Transformer (base-sized)
- [EfficientNet](https://huggingface.co/google/efficientnet-b0): EfficientNet (b0-b7 sized)
- [Focalnet](https://huggingface.co/microsoft/focalnet-tiny): FocalNet (tiny-sized)
- [Mobilevit](https://huggingface.co/apple/mobilevit-xx-small): Apple's MobileViT xx-small
- [Mobilevit2](https://huggingface.co/apple/mobilevitv2-1.0-imagenet1k-256): Apple's MobileViTv2
- [Pvt](https://huggingface.co/Zetatech/pvt-tiny-224): Pyramid Vision Transformer (tiny-sized)
- [Swin](https://huggingface.co/microsoft/swin-tiny-patch4-window7-224): Swin Transformer (tiny-sized)

### Audio Models
#### ASR (Automatic Speech Recognition)
- [Whisper](https://huggingface.co/openai/whisper-tiny): OpenAI's `Whisper` and its variants

#### Speech text-to-text (Automatic Speech Recognition)
- 💡[**NEW**] [Granite Speech](https://huggingface.co/ibm-granite/granite-speech-3.3-2b): `granite-speech-3.3-2b` and its variants
- 💡[**NEW**] [Voxtral](https://huggingface.co/mistralai/Voxtral-Mini-3B-2507): Mistral's newest speech/text-to-text model

*📌 Note: This list is continuously expanding. As we continue to expand support, more models will be added.*

## 🚀 Benchmarks on Mobile Devices

The following benchmarks show example **decode performance** (tokens/sec) across Android and iOS devices for popular edge LLMs.

| Model | Samsung Galaxy S22 5G<br/>(Android 13) | Samsung Galaxy S22 Ultra 5G<br/>(Android 14) | iPhone 15<br/>(iOS 18.0) | iPhone 15 Plus<br/>(iOS 17.4.1) | iPhone 15 Pro<br/>(iOS 18.4.1) |
|-------|:---:|:---:|:---:|:---:|:---:|
| [**SmolLM2-135M**](https://tinyurl.com/25ud3th8) | 202.28 | 202.61 | 7.47 | 6.43 | 29.64 |
| [**Qwen3-0.6B**](https://tinyurl.com/35946h8b) | 59.16 | 56.49 | 7.05 | 5.48 | 17.99 |
| [**google/gemma-3-1b-it**](https://tinyurl.com/4d8pezpv) | 25.07 | 23.89 | 21.51 | 21.33 | 17.8 |
| [**Llama-3.2-1B**](https://tinyurl.com/bddjewau) | 44.91 | 37.39 | 11.04 | 8.93 | 25.78 |
| [**OLMo-1B**](https://tinyurl.com/4runxesd) | 44.98 | 38.22 | 14.49 | 8.72 | 20.24 |

> 📊 **View Live Benchmarks**: Explore comprehensive performance data, compare models across devices, and track performance trends over time on the [ExecuTorch Benchmark Dashboard](https://hud.pytorch.org/benchmark/llms?repoName=pytorch%2Fexecutorch).

> Performance measured with custom SDPA, KV-cache optimization, and 8da4w quantization. Results may vary based on device conditions and prompt characteristics.


## 🛠️ Advanced Usage

Check our [ExecuTorch GitHub repo](https://github.com/pytorch/executorch) directly for:
- More backends and performance optimization options
- Deployment guides for Android, iOS, and embedded devices
- Additional examples and benchmarks

## 🤝 Contributing

We love your input! We want to make contributing to Optimum ExecuTorch as easy and transparent as possible. Check out our:

- [Contributing Guidelines](https://github.com/huggingface/optimum/blob/main/CONTRIBUTING.md)
- [Code of Conduct](https://github.com/huggingface/optimum/blob/main/CODE_OF_CONDUCT.md)

## 📝 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](https://github.com/huggingface/optimum/blob/main/LICENSE) file for details.

## 📫 Get in Touch

- Report bugs through [GitHub Issues](https://github.com/huggingface/optimum-executorch/issues)
