from ga_model_utils import run_with_coreml, run_with_et

output_dir_base = f"/Users/scroy/Desktop/ga-model-exports"

models = [
    "smollm",
    "efficientnet",
    "resnet",
    "sam_vision_encoder",
    "vit",
    "whisper",
    # "yolos", # takes a while to load
    "yolos2",
]
from collections import defaultdict
submodels = defaultdict(lambda: ["model"])
submodels["whisper"] = ["decoder", "encoder"]


for model in models:
    output_dir = f"{output_dir_base}/{model}"
    for submodel in submodels[model]:
       
        print(f"\n\nRunning {model}/{submodel}")
        try:
            run_with_coreml(f"{output_dir}/{submodel}.mlpackage")
        except Exception as e:
            print(f"Runtime error: {e}")
        
        try:
            run_with_et(f"{output_dir}/{submodel}.pte")
        except Exception as e:
            print(f"Runtime error: {e}")
