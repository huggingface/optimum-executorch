from optimum.exporters.executorch import main_export
from ga_model_utils import get_model_exporter, lower_with_coreml, lower_with_et
import os

et_optimum_ga_models = {
    "smollm": {"model_name_or_path": "HuggingFaceTB/SmolLM2-135M-Instruct", "task": "text-generation"},
    "vit": {"model_name_or_path": "google/vit-base-patch16-224", "task": "image-classification"},
    "efficientnet": {"model_name_or_path": "google/efficientnet-b0", "task": "image-classification"},
    "resnet": {"model_name_or_path": "microsoft/resnet-50", "task": "image-classification"},
    "whisper": {"model_name_or_path": "openai/whisper-tiny", "task": "automatic-speech-recognition"},
    "yolos": {"model_name_or_path": "hustvl/yolos-tiny", "task": "object-detection"},
}
other_ga_models = [
    "sam_vision_encoder", "yolos2",
]

output_dir_base = f"/Users/scroy/Desktop/ga-model-exports"

settings = {
    "take_over_mutable_buffer": False,
    "minimum_ios_deployment_target": "18",
    "model_type": "modelc",
}


def export_coreml_standalone(model, kwargs, output_dir):
    try:
        main_export(
            model_name_or_path=kwargs["model_name_or_path"],
            task=kwargs["task"],
            output_dir=output_dir,
            recipe="coreml_standalone",
        )
    except Exception as e:
        log_path = os.path.join(output_dir, "coreml_exception.txt")
        with open(log_path, "w") as f:
            f.write(str(e) + "\n")

def export_coreml_et(model, kwargs, output_dir):
    try:
        main_export(
            model_name_or_path=kwargs["model_name_or_path"],
            task=kwargs["task"],
            output_dir=output_dir,
            recipe="coreml",
        )
    except Exception as e:
        log_path = os.path.join(output_dir, "executorch_exception.txt")
        with open(log_path, "w") as f:
            f.write(str(e) + "\n")
    
    for setting, value in settings.items():
        try:
            main_export(
                model_name_or_path=kwargs["model_name_or_path"],
                task=kwargs["task"],
                output_dir=f"{output_dir}/{setting}-{settings[setting]}",
                recipe="coreml",
                **{"recipe_kwargs": {setting: value}},
            )
        except Exception as e:
            log_path = os.path.join(output_dir, f"executorch_exception_{setting}-{value}.txt")
            with open(log_path, "w") as f:
                f.write(str(e))


for model, kwargs in et_optimum_ga_models.items():
    output_dir = f"{output_dir_base}/{model}"
    export_coreml_standalone(model, kwargs, output_dir)
    export_coreml_et(model, kwargs, output_dir)

for model in other_ga_models:
    ep = get_model_exporter(model)()
    output_dir = f"{output_dir_base}/{model}"
    export_coreml_standalone(model, kwargs, output_dir)
    export_coreml_et(model, kwargs, output_dir)
