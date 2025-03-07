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
pip install .
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
bash ./install_executorch.sh
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
model_id = "meta-llama/Llama-3.2-1B"
model = ExecuTorchModelForCausalLM.from_pretrained(model_id, recipe="xnnpack")

# Generate text right away
tokenizer = AutoTokenizer.from_pretrained(model_id)
generated_text = model.text_generation(
    tokenizer=tokenizer,
    prompt="Simply put, the theory of relativity states that",
    max_seq_len=128
)
print(generated_text)
```

> **Note:** If an ExecuTorch model is already cached on the Hugging Face Hub, the API will automatically skip the export step and load the cached `.pte` file. To test this, replace the `model_id` in the example above with `"executorch-community/SmolLM2-135M"`, where the `.pte` file is pre-cached. Additionally, the `.pte` file can be directly associated with the eager model, as demonstrated in this [example](https://huggingface.co/optimum-internal-testing/tiny-random-llama/tree/executorch).


### Option 2: Export and Load Separately

#### Step 1: Export your model
Use the CLI tool to convert your model to ExecuTorch format:
```
optimum-cli export executorch \
    --model "meta-llama/Llama-3.2-1B" \
    --task "text-generation" \
    --recipe "xnnpack" \
    --output_dir="meta_llama3_2_1b"
```

#### Step 2: Load and run inference
Use the exported model for text generation:
```python
from optimum.executorch import ExecuTorchModelForCausalLM
from transformers import AutoTokenizer

# Load the exported model
model = ExecuTorchModelForCausalLM.from_pretrained("./meta_llama3_2_1b")

# Initialize tokenizer and generate text
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
generated_text = model.text_generation(
    tokenizer=tokenizer,
    prompt="Simply put, the theory of relativity states that",
    max_seq_len=128
)
print(generated_text)
```

## Supported Models and Backend

**Optimum-ExecuTorch** currently supports the following transformer models:

- [meta-llama/Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B) and its variants
- [HuggingFaceTB/SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M) and its variants
- [Qwen/Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B) and its variants
- [deepseek-ai/DeepSeek-R1-Distill-Llama-8B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B) and its variants
- [google/gemma-2-2b](https://huggingface.co/google/gemma-2-2b) and its variants
- [allenai/OLMo-1B-hf](https://huggingface.co/allenai/OLMo-1B-hf) and its variants

*Note: This list is continuously expanding. As we continue to expand support, more models and variants will be added.*

**Supported Backend:**

Currently, **Optimum-ExecuTorch** supports only the [XNNPACK Backend](https://pytorch.org/executorch/main/backends-xnnpack.html) for efficient CPU execution on mobile devices. Quantization support for XNNPACK is planned to be added shortly.

For a comprehensive overview of all backends supported by ExecuTorch, please refer to the [ExecuTorch Backend Overview](https://pytorch.org/executorch/main/backends-overview.html).


## 🛠️ Advanced Usage

Check our [ExecuTorch GitHub repo](https://github.com/pytorch/executorch) directly for:
- More backends and performance optimization options
- Deployment guides for Android, iOS, and embedded devices
- Additional examples and benchmarks

## 🤝 Contributing

We love your input! We want to make contributing to Optimum ExecuTorch as easy and transparent as possible. Check out our:

- [Contributing Guidelines](CONTRIBUTING.md)
- [Code of Conduct](CODE_OF_CONDUCT.md)

## 📝 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 📫 Get in Touch

- Report bugs through [GitHub Issues](https://github.com/huggingface/optimum-executorch/issues)
