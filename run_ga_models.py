from ga_model_utils import run_with_coreml, run_with_et
from ga_model_configs import et_optimum_ga_models, submodels
import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process export options.")
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Path to the output directory"
    )
    args = parser.parse_args()

    output_dir_base = args.model_dir

    for model in et_optimum_ga_models:
        output_dir = f"{output_dir_base}/{model}"
        for submodel in submodels[model]:
        
            print(f"\n\nRunning {model}/{submodel}")
            try:
                print("CoreML standalone")
                model_path = f"{output_dir}/coreml_standalone/{submodel}.mlpackage"
                if os.path.exists(model_path):
                    run_with_coreml(model_path)
                else:
                    print("No model found")
            except Exception as e:
                print(f"Runtime error: {e}")
            
            try:
                print("ET CoreML backend")
                model_path = f"{output_dir}/coreml/{submodel}.pte"
                if os.path.exists(model_path):
                    run_with_et(model_path)
                else:
                    print("No model found")
            except Exception as e:
                print(f"Runtime error: {e}")

            try:
                print("ET XNNPACK backend")
                model_path = f"{output_dir}/xnnpack/{submodel}.pte"
                if os.path.exists(model_path):
                    run_with_et(model_path)
                else:
                    print("No model found")
            except Exception as e:
                print(f"Runtime error: {e}")
