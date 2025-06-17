# Temporary spot for exporters that live outside optimum-executorch

from typing import Callable, Dict, Optional
from torch.export import ExportedProgram
import torch
from transformers import AutoModelForImageClassification, AutoProcessor, YolosForObjectDetection, AutoImageProcessor, SamProcessor, SamModel, Wav2Vec2Model, WhisperProcessor, WhisperForConditionalGeneration
from executorch.backends.apple.coreml.compiler import CoreMLBackend
from executorch.backends.apple.coreml.partition import CoreMLPartitioner
from executorch.exir import to_edge_transform_and_lower
import coremltools as ct
from PIL import Image
from executorch.runtime import Runtime
import numpy as np
import time
import tqdm
import os

MODEL_EXPORTERS: Dict[str, Callable[[], ExportedProgram]] = {}
def register_model_exporter(name: str):
    def decorator(func: Callable[[], ExportedProgram]):
        if name in MODEL_EXPORTERS:
            raise ValueError(f"Cannot register duplicate model exporter ({name})")
        MODEL_EXPORTERS[name] = func
        return func
    return decorator

def get_model_exporter(name: str) -> Callable[[], ExportedProgram]:
    try:
        return MODEL_EXPORTERS[name]
    except KeyError:
        raise ValueError(f"No model exporter registered under name '{name}'")

@register_model_exporter("sam_vision_encoder")
def _() -> ExportedProgram:
    # Just do the vision encoder here
    model_id = "facebook/sam-vit-base"
    processor = SamProcessor.from_pretrained(model_id)
    model = SamModel.from_pretrained(model_id)
    model.eval()

    # Create dummy image (e.g., 512x512 RGB)
    dummy_image = Image.fromarray((np.random.rand(512, 512, 3) * 255).astype(np.uint8))

    # Create a dummy point prompt (normalized coordinates in [0, 1])
    dummy_input_points = torch.tensor([[[0.5, 0.5]]])  # Shape: [batch, num_points, 2]
    dummy_input_labels = torch.tensor([[1]])           # 1 = foreground

    # Preprocess input
    inputs = processor(
        dummy_image,
        input_points=dummy_input_points,
        input_labels=dummy_input_labels,
        return_tensors="pt"
    )

    model = model.vision_encoder
    example_inputs = (inputs["pixel_values"],)
    exported_model = torch.export.export(model, args=example_inputs)
    return exported_model


@register_model_exporter("yolos2")
def _() -> ExportedProgram:
    # Load pretrained YOLOS model and image processor
    model_id = "hustvl/yolos-tiny"
    model = YolosForObjectDetection.from_pretrained(model_id)
    processor = AutoImageProcessor.from_pretrained(model_id)
    model.eval()

    # Create a dummy RGB image (224x224 or any size; YOLOS resizes internally)
    dummy_image = Image.fromarray((np.random.rand(224, 224, 3) * 255).astype(np.uint8))

    # Preprocess image
    inputs = processor(images=dummy_image, return_tensors="pt")
    assert len(inputs) == 1
    example_inputs = (inputs["pixel_values"],)

    exported_model = torch.export.export(model, args=example_inputs)
    return exported_model


@register_model_exporter("whisper2")
def _() -> ExportedProgram:
    model_id = "openai/whisper-small"
    model = WhisperForConditionalGeneration.from_pretrained(model_id)
    processor = WhisperProcessor.from_pretrained(model_id)
    model.eval()

    # Create dummy audio input: 30 seconds of mono audio sampled at 16kHz
    dummy_audio = np.random.rand(480000).astype(np.float32)  # 30s * 16kHz

    # Process dummy audio to model inputs
    inputs = processor(dummy_audio, sampling_rate=16000, return_tensors="pt")

    bos_token_id = processor.tokenizer.bos_token_id  
    decoder_input_ids = torch.tensor([[bos_token_id]])

    # Whisper model expects input_values and optional decoder_input_ids
    assert len(inputs) == 1
    example_inputs = (inputs["input_features"], decoder_input_ids)
    class WhisperExportWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, input_features, decoder_input_ids):
            outputs = self.model(input_features=input_features, decoder_input_ids=decoder_input_ids)
            # return logits only for simplicity
            return outputs.logits
    
    wrapped_model = WhisperExportWrapper(model)
    exported_model = torch.export.export(wrapped_model, args=example_inputs)
    return exported_model



@register_model_exporter("wave2vec2")
def _() -> ExportedProgram:
    model_id = "facebook/wav2vec2-base"

    # Load model and processor
    model = Wav2Vec2Model.from_pretrained(model_id)
    processor = AutoProcessor.from_pretrained(model_id)
    model.eval()

    # Create dummy audio input: 1 second of audio at 16kHz
    dummy_waveform = np.random.rand(16000).astype(np.float32)
    inputs = processor(dummy_waveform, return_tensors="pt", sampling_rate=16000)

    # Prepare example input
    example_inputs = (inputs["input_values"],)

    # Export the model
    exported_model = torch.export.export(model, args=example_inputs)

    return exported_model


def lower_with_et(exported_program, filename=None):
    assert filename is not None
    parent_dir = os.path.dirname(filename)
    os.makedirs(parent_dir, exist_ok=True)

    et_program = to_edge_transform_and_lower(
        exported_program,
        partitioner=[CoreMLPartitioner()],
    ).to_executorch()
    with open(filename, "wb") as file:
        et_program.write_to_file(file)

def lower_with_coreml(exported_program, filename=None):
    assert filename is not None
    exported_program = exported_program.run_decompositions({})
    ml_model = ct.convert(exported_program)
    ml_model.save(filename)

def run_with_et(filename, n_iters=50):
    runtime = Runtime.get()

    program = runtime.load_program(filename)
    method = program.load_method("forward")
    dtype_lookup = {6: torch.float32, 4: torch.int64}
    inputs = []
    for i in range(method.metadata.num_inputs()):
        t_metadata = method.metadata.input_tensor_meta(i)
        if t_metadata.dtype() in dtype_lookup:
            dtype = dtype_lookup[t_metadata.dtype()]
            if dtype in [torch.int64]:
                inputs.append(torch.randint(0, 100, t_metadata.sizes(), dtype=dtype))
            else:
                inputs.append(torch.rand(t_metadata.sizes(), dtype=dtype))
        else:
            raise Exception(f"Unsupported input type: {t_metadata.dtype()} in {t_metadata}")

    start = time.time()
    for _ in tqdm.tqdm(range(n_iters), total=n_iters):
        outputs: List[torch.Tensor] = method.execute(inputs)
    end = time.time()
    ms_per_iter = (end - start) / n_iters * 1000
    print(f"ExecuTorch model execution time (ms): {ms_per_iter:.6f}")


def run_with_coreml(filename, n_iters=50):
    ml_model = ct.models.MLModel(filename)
    spec = ml_model.get_spec()
    inputs = {}
    for inp in spec.description.input:
        shape = []
        if inp.type.WhichOneof("Type") == "multiArrayType":
            array_type = inp.type.multiArrayType
            shape = [int(dim) for dim in array_type.shape]
            dtype = np.float32 if array_type.dataType == array_type.FLOAT32 else np.float64
            inputs[inp.name] = np.random.rand(*shape).astype(dtype)
        else:
            raise Exception(f"Unsupported input type: {inp.type.WhichOneof('Type')}")

    start = time.time()
    for _ in tqdm.tqdm(range(n_iters), total=n_iters):
        ml_model.predict(inputs)
    end = time.time()
    ms_per_iter = (end - start) / n_iters * 1000
    print(f"CoreML model execution time (ms): {ms_per_iter:.6f}")
