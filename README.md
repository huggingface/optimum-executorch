<div align="center">

<img src="https://huggingface.co/datasets/optimum/documentation-images/resolve/main/executorch/logo/optimum-executorch.png" width=80%>

# 🤗 Optimum ExecuTorch

**Optimize and deploy Hugging Face models with ExecuTorch**

[Documentation](https://huggingface.co/docs/optimum/index) | [ExecuTorch](https://github.com/pytorch/executorch) | [Hugging Face](https://huggingface.co/)

</div>

## 🚀 Overview

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
pip install '.[tests]'
```

- 🔜 Install from pypi coming soon...

### [Optional] 3. Install dependencies in dev mode
You can install `executorch` and `transformers` from source, where you can access new ExecuTorch
compatilbe models from `transformers` and new features from `executorch` as both repos are under
rapid deployment.

Follow these steps manually:

#### 3.1. Clone and Install ExecuTorch from Source
From the root directory where `optimum-executorch` is cloned:
```
# Clone the ExecuTorch repository
git clone https://github.com/pytorch/executorch.git
cd executorch
# Checkout the stable branch to ensure stability
git checkout viable/strict
# Install ExecuTorch
python ./install_executorch.py
cd ..
```

#### 3.2. Clone and Install Transformers from Source
From the root directory where `optimum-executorch` is cloned:
```
# Clone the Transformers repository
git clone https://github.com/huggingface/transformers.git
cd transformers
# Install Transformers in editable mode
pip install -e .
cd ..
```

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
    **{"qlinear": True},  # quantize linear layers with 8da4w
)

# Generate text right away
tokenizer = AutoTokenizer.from_pretrained(model_id)
generated_text = model.text_generation(
    tokenizer=tokenizer,
    prompt="Once upon a time",
    max_seq_len=32,
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
    --output_dir="hf_smollm2" \
    --use_custom_sdpa \
    --qlinear
```
Explore the various export options by running the command: `optimum-cli export executorch --help`

#### Step 2: Load and run inference
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

## Supported Optimizations

### Custom Operators
Supported using [custom SDPA](https://github.com/pytorch/executorch/blob/a4322c71c3a97e79e0454a8223db214b010f1193/extension/llm/README.md?plain=1#L40) with Hugging Face Transformers, boosting performance by 3x compared to default SDPA, based on tests with `HuggingFaceTB/SmolLM2-135M`.

### Backends Delegation
Currently, **Optimum-ExecuTorch** supports the [XNNPACK Backend](https://pytorch.org/executorch/main/backends-xnnpack.html) with [custom SDPA](https://github.com/pytorch/executorch/blob/a4322c71c3a97e79e0454a8223db214b010f1193/extension/llm/README.md?plain=1#L40) for efficient execution on mobile CPUs.

For a comprehensive overview of all backends supported by ExecuTorch, please refer to the [ExecuTorch Backend Overview](https://pytorch.org/executorch/main/backends-overview.html).

### Quantization
We currently support Post-Training Quantization (PTQ) for linear layers using int8 dynamic per-token activations and int4 grouped per-channel weights (aka `8da4w`), as well as int8 channelwise embedding quantization.

🚀 Stay tuned as more optimizations and performance enhancements are coming soon!


## Supported Models

The following models have been successfully tested with Executorch. For details on the specific optimizations supported and how to use them for each model, please consult their respective test files in the [`tests/models/`](https://github.com/huggingface/optimum-executorch/tree/main/tests/models) directory.

### Text Models
We currently support a wide range of popular transformer models, including encoder-only, decoder-only, and encoder-decoder architectures, as well as models specialized for various tasks like text generation, translation, summarization, and mask prediction, etc. These models reflect the current trends and popularity across the Hugging Face community:
#### Encoder-only models
- [Albert](https://huggingface.co/albert/albert-base-v2): `albert-base-v2` and its variants
- [Bert](https://huggingface.co/google-bert/bert-base-uncased): Google's `bert-base-uncased` and its variants
- [Distilbert](https://huggingface.co/distilbert/distilbert-base-uncased): `distilbert-base-uncased` and its variants
- [Eurobert](https://huggingface.co/EuroBERT/EuroBERT-210m): `EuroBERT-210m` and its variants
- [Roberta](https://huggingface.co/FacebookAI/xlm-roberta-base): FacebookAI's `xlm-roberta-base` and its variants
#### Decoder-only models
- [Gemma](https://huggingface.co/google/gemma-2b): `Gemma-2b` and its variants
- [Gemma2](https://huggingface.co/google/gemma-2-2b): `Gemma-2-2b` and its variants
- [Gemma3](https://huggingface.co/google/gemma-3-1b-it): `Gemma-3-1b` and its variants *(requires install latest `transformers (4.52.0.dev0)` manually from source)*
- [Llama](https://huggingface.co/meta-llama/Llama-3.2-1B): `Llama-3.2-1B` and its variants
- [Qwen2](https://huggingface.co/Qwen/Qwen2.5-0.5B): `Qwen2.5-0.5B` and its variants
- [Qwen3](https://huggingface.co/Qwen/Qwen3-0.6B): `Qwen3-0.6B` and its variants
- [Olmo](https://huggingface.co/allenai/OLMo-1B-hf): `OLMo-1B-hf` and its variants
- [Phi4](https://huggingface.co/microsoft/Phi-4-mini-instruct): `Phi-4-mini-instruct` and its variants
- [Smollm](https://huggingface.co/HuggingFaceTB/SmolLM2-135M): 🤗 `SmolLM2-135M` and its variants
#### Encoder-decoder models
- [T5](https://huggingface.co/google-t5/t5-small): Google's `T5` and its variants

### Vision Models
#### Encoder-only models
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
#### Encoder-decoder models
- [Whisper](https://huggingface.co/openai/whisper-tiny): OpenAI's `Whisper` and its variants

*📌 Note: This list is continuously expanding. As we continue to expand support, more models will be added.*


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
