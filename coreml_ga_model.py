# from optimum.executorch import ExecuTorchModelForCausalLM
# from transformers import AutoTokenizer

from optimum.exporters.executorch import main_export

# # Load and export the model on-the-fly
# model_id = "HuggingFaceTB/SmolLM2-135M-Instruct"
# model = ExecuTorchModelForCausalLM.from_pretrained(
#     model_id,
#     recipe="coreml",
#     # attn_implementation="custom_sdpa",  # Use custom SDPA implementation for better performance
#     # **{"qlinear": True},  # quantize linear layers with 8da4w
# )

# # Generate text right away
# tokenizer = AutoTokenizer.from_pretrained(model_id)
# generated_text = model.text_generation(
#     tokenizer=tokenizer,
#     prompt="Once upon a time",
#     max_seq_len=32,
# )
# print(generated_text)



# import torch
# from PIL import Image
# from transformers import AutoModelForImageClassification, AutoProcessor, YolosForObjectDetection, AutoImageProcessor, SamProcessor, SamModel, Wav2Vec2Model, WhisperProcessor, WhisperForConditionalGeneration
# import numpy as np
# from torch.export import ExportedProgram
# from typing import Callable, Dict, Optional


# # Registry to store exporter functions
# MODEL_EXPORTERS: Dict[str, Callable[[], ExportedProgram]] = {}
# def register_model_exporter(name: str):
#     def decorator(func: Callable[[], ExportedProgram]):
#         if name in MODEL_EXPORTERS:
#             raise ValueError(f"Cannot register duplicate model exporter ({name})")
#         MODEL_EXPORTERS[name] = func
#         return func
#     return decorator

# def get_model_exporter(name: str) -> Callable[[], ExportedProgram]:
#     try:
#         return MODEL_EXPORTERS[name]
#     except KeyError:
#         raise ValueError(f"No model exporter registered under name '{name}'")


# @register_model_exporter("resnet")
# def _() -> ExportedProgram:
#     model_id = "microsoft/resnet-50"
#     model = AutoModelForImageClassification.from_pretrained(model_id)
#     processor = AutoProcessor.from_pretrained(model_id)
#     model.eval()

#     # Create a dummy image (e.g., 224x224 RGB)
#     dummy_image = Image.fromarray((np.random.rand(224, 224, 3) * 255).astype(np.uint8))
#     inputs = processor(images=dummy_image, return_tensors="pt")
#     assert len(inputs) == 1
#     example_inputs = (inputs["pixel_values"],)

#     exported_model = torch.export.export(model, args=example_inputs)
#     return exported_model

# @register_model_exporter("vit")
# def _() -> ExportedProgram:
#     # Load pretrained ViT model
#     model_id = "google/vit-base-patch16-224"
#     model = AutoModelForImageClassification.from_pretrained(model_id)
#     processor = AutoProcessor.from_pretrained(model_id)

#     dummy_image = Image.fromarray((np.random.rand(224, 224, 3) * 255).astype(np.uint8))
#     inputs = processor(images=dummy_image, return_tensors="pt")
#     assert len(inputs) == 1
#     example_inputs = (inputs["pixel_values"],)

#     exported_model = torch.export.export(model, args=example_inputs)
#     return exported_model

# @register_model_exporter("yolos")
# def _() -> ExportedProgram:
#     # Load pretrained YOLOS model and image processor
#     model_id = "hustvl/yolos-tiny"
#     model = YolosForObjectDetection.from_pretrained(model_id)
#     processor = AutoImageProcessor.from_pretrained(model_id)
#     model.eval()

#     # Create a dummy RGB image (224x224 or any size; YOLOS resizes internally)
#     dummy_image = Image.fromarray((np.random.rand(224, 224, 3) * 255).astype(np.uint8))

#     # Preprocess image
#     inputs = processor(images=dummy_image, return_tensors="pt")
#     assert len(inputs) == 1
#     example_inputs = (inputs["pixel_values"],)

#     exported_model = torch.export.export(model, args=example_inputs)
#     return exported_model

# @register_model_exporter("sam_vision_encoder")
# def _() -> ExportedProgram:
#     # I cannot get sam to export, so I just do the vision encoder here
#     model_id = "facebook/sam-vit-base"
#     processor = SamProcessor.from_pretrained(model_id)
#     model = SamModel.from_pretrained(model_id)
#     model.eval()

#     # Create dummy image (e.g., 512x512 RGB)
#     dummy_image = Image.fromarray((np.random.rand(512, 512, 3) * 255).astype(np.uint8))

#     # Create a dummy point prompt (normalized coordinates in [0, 1])
#     dummy_input_points = torch.tensor([[[0.5, 0.5]]])  # Shape: [batch, num_points, 2]
#     dummy_input_labels = torch.tensor([[1]])           # 1 = foreground

#     # Preprocess input
#     inputs = processor(
#         dummy_image,
#         input_points=dummy_input_points,
#         input_labels=dummy_input_labels,
#         return_tensors="pt"
#     )

#     model = model.vision_encoder
#     example_inputs = (inputs["pixel_values"],)
#     exported_model = torch.export.export(model, args=example_inputs)
#     return exported_model

# @register_model_exporter("wave2vec2")
# def _() -> ExportedProgram:
#     model_id = "facebook/wav2vec2-base"

#     # Load model and processor
#     model = Wav2Vec2Model.from_pretrained(model_id)
#     processor = AutoProcessor.from_pretrained(model_id)
#     model.eval()

#     # Create dummy audio input: 1 second of audio at 16kHz
#     dummy_waveform = np.random.rand(16000).astype(np.float32)
#     inputs = processor(dummy_waveform, return_tensors="pt", sampling_rate=16000)

#     # Prepare example input
#     example_inputs = (inputs["input_values"],)

#     # Export the model
#     exported_model = torch.export.export(model, args=example_inputs)

#     return exported_model

# @register_model_exporter("whisper")
# def _() -> ExportedProgram:
#     model_id = "openai/whisper-tiny"
#     model = WhisperForConditionalGeneration.from_pretrained(model_id)
#     processor = WhisperProcessor.from_pretrained(model_id)
#     model.eval()

#     # Create dummy audio input: 30 seconds of mono audio sampled at 16kHz
#     dummy_audio = np.random.rand(480000).astype(np.float32)  # 30s * 16kHz

#     # Process dummy audio to model inputs
#     inputs = processor(dummy_audio, sampling_rate=16000, return_tensors="pt")

#     bos_token_id = processor.tokenizer.bos_token_id  
#     decoder_input_ids = torch.tensor([[bos_token_id]])

#     # Whisper model expects input_values and optional decoder_input_ids
#     assert len(inputs) == 1
#     example_inputs = (inputs["input_features"], decoder_input_ids)
#     class WhisperExportWrapper(torch.nn.Module):
#         def __init__(self, model):
#             super().__init__()
#             self.model = model

#         def forward(self, input_features, decoder_input_ids):
#             outputs = self.model(input_features=input_features, decoder_input_ids=decoder_input_ids)
#             # return logits only for simplicity
#             return outputs.logits

#     wrapped_model = WhisperExportWrapper(model)
#     wrapped_model.eval()

#     # Export the model
#     exported_model = torch.export.export(wrapped_model, example_inputs)


   

#     # Export the model using torch.export
#     # exported_model = torch.export.export(model, args=example_inputs)
#     return exported_model



# from executorch.backends.apple.coreml.compiler import CoreMLBackend
# from executorch.backends.apple.coreml.partition import CoreMLPartitioner
# from executorch.exir import to_edge_transform_and_lower
# import coremltools as ct
# import subprocess


# import torch
from executorch.runtime import Runtime
# from typing import List

# import tqdm
# import time




# def lower_with_et(exported_program, filename=None, extract_coreml_model_script: Optional[str] = None, minimum_deployment_target=ct.target.iOS15):
#     assert filename is not None
#     et_program = to_edge_transform_and_lower(
#         exported_program,
#         partitioner=[CoreMLPartitioner(
#             compile_specs=CoreMLBackend.generate_compile_specs(
#                         minimum_deployment_target=minimum_deployment_target,
#                     ),
#         )],
#     ).to_executorch()
#     with open(filename, "wb") as file:
#         et_program.write_to_file(file)

#     if extract_coreml_model_script is not None:
#         subprocess.run([
#             "python",
#             extract_coreml_model_script,
#             "-m",
#             filename
#         ])







# def lower_with_coreml(exported_program, filename=None, minimum_deployment_target=ct.target.iOS15):
#     assert filename is not None
#     exported_program = exported_program.run_decompositions({})
#     ml_model = ct.convert(exported_program, minimum_deployment_target=minimum_deployment_target)
#     ml_model.save(filename)



# # # exported_model = get_model_exporter("resnet")()
# name = "whisper"
# et_model = f"{name}_et.pte"
# coreml_model = f"{name}_coreml.mlpackage"
# exported_model = get_model_exporter(name)()
# # lower_with_coreml(exported_model, coreml_model)

# # # raise Exception("TODO")
# # # filename = "resnet_et.pte"
# lower_with_et(exported_model, et_model, extract_coreml_model_script="/Users/scroy/repos/executorch/examples/apple/coreml/scripts/extract_coreml_models.py") 

# n_iters = 1
# run_with_et(et_model, exported_model.example_inputs, n_iters)
# # # run_with_coreml(coreml_model, exported_model.example_inputs, n_iters)
# # # lower_with_coreml(exported_model, "resnet_coreml.mlpackage")


ga_models = {
    # "vit": {"model_name_or_path": "google/vit-base-patch16-224", "task": "image-classification"},
    # "efficientnet": {"model_name_or_path": "google/efficientnet-b0", "task": "image-classification"},
    "whisper": {"model_name_or_path": "openai/whisper-tiny", "task": "automatic-speech-recognition"},
}
for model, kwargs in ga_models.items():
    output_dir = f"/Users/scroy/Desktop/model-exports/{model}"
    print("Exporting to CoreML standalone")
    try:
        main_export(
            model_name_or_path=kwargs["model_name_or_path"],
            task=kwargs["task"],
            output_dir=output_dir,
            recipe="coreml_standalone",
        )
    except Exception as e:
        print(f"Failed to export {model} to CoreML standalone: {e}")

    print("Exporting to CoreML")
    try:
        main_export(
            model_name_or_path=kwargs["model_name_or_path"],
            task=kwargs["task"],
            output_dir=output_dir,
            recipe="coreml",
        )
    except Exception as e:
        print(f"Failed to export {model} to CoreML: {e}")


# main_export(
#     model_name_or_path="google/vit-base-patch16-224",
#     task="image-classification",
#     output_dir="/Users/scroy/Desktop/model-exports",
#     recipe="coreml_standalone",
# )

# main_export(
#     model_name_or_path="google/vit-base-patch16-224",
#     task="image-classification",
#     output_dir="/Users/scroy/Desktop/model-exports",
#     recipe="coreml",
# )


import numpy as np
import coremltools as ct
import time
import tqdm

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


# run_with_coreml("/Users/scroy/Desktop/model-exports/model.mlpackage")

import torch
def run_with_et(filename, n_iters=50):
    runtime = Runtime.get()

    program = runtime.load_program(filename)
    method = program.load_method("forward")
    dtype_lookup = {6: torch.float32}
    inputs = []
    for i in range(method.metadata.num_inputs()):
        t_metadata = method.metadata.input_tensor_meta(i)
        if t_metadata.dtype() in dtype_lookup:
            inputs.append(torch.rand(t_metadata.sizes(), dtype=dtype_lookup[t_metadata.dtype()]))
        else:
            raise Exception(f"Unsupported input type: {t_metadata.dtype()} in {t_metadata}")

    start = time.time()
    for _ in tqdm.tqdm(range(n_iters), total=n_iters):
        outputs: List[torch.Tensor] = method.execute(inputs)
    end = time.time()
    ms_per_iter = (end - start) / n_iters * 1000
    print(f"ExecuTorch model execution time (ms): {ms_per_iter:.6f}")


# run_with_et("/Users/scroy/Desktop/model-exports/model.pte")
