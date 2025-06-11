
et_optimum_ga_models = {
    "smollm": {"model_name_or_path": "HuggingFaceTB/SmolLM2-135M-Instruct", "task": "text-generation"},
    "smollm_mutable_buffer_false": {"model_name_or_path": "HuggingFaceTB/SmolLM2-135M-Instruct", "task": "text-generation", "recipe_kwargs": {"take_over_mutable_buffer": False}},
    "smollm_mutable_buffer_false_fp32": {"model_name_or_path": "HuggingFaceTB/SmolLM2-135M-Instruct", "task": "text-generation", "recipe_kwargs": {"take_over_mutable_buffer": False, "compute_precision": "fp32"}},
    "vit": {"model_name_or_path": "google/vit-base-patch16-224", "task": "image-classification"},
    "efficientnet": {"model_name_or_path": "google/efficientnet-b0", "task": "image-classification"},
    "efficientnet_modelc": {"model_name_or_path": "google/efficientnet-b0", "task": "image-classification", "recipe_kwargs": {"model_type": "modelc"}},
    "efficientnet-quantize-ios17": {"model_name_or_path": "google/efficientnet-b0", "task": "image-classification", "recipe_kwargs": {"quantize": True,  "minimum_ios_deployment_target": "17"}},
    "resnet": {"model_name_or_path": "microsoft/resnet-50", "task": "image-classification"},
    "resnet_ios18": {"model_name_or_path": "microsoft/resnet-50", "task": "image-classification", "recipe_kwargs": {"minimum_ios_deployment_target": "18"}},
    "whisper": {"model_name_or_path": "openai/whisper-tiny", "task": "automatic-speech-recognition"},
    "whisper2": {"is_optimum": False},
    "yolos": {"model_name_or_path": "hustvl/yolos-tiny", "task": "object-detection"},
    "yolos-quantize": {"model_name_or_path": "hustvl/yolos-tiny", "task": "object-detection", "recipe_kwargs": {"quantize": True}},
    "yolos-quantize-ios17": {"model_name_or_path": "hustvl/yolos-tiny", "task": "object-detection", "recipe_kwargs": {"quantize": True, "minimum_ios_deployment_target": "17"}},
    "sam_vision_encoder": {"is_optimum": False},
    "yolos2": {"is_optimum": False},
}


from collections import defaultdict
submodels = defaultdict(lambda: ["model"])
submodels["whisper"] = ["decoder", "encoder"]
