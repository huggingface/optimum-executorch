from optimum.exporters.executorch import main_export
from ga_model_utils import get_model_exporter, lower_with_coreml, lower_with_et
from ga_model_configs import et_optimum_ga_models
import os
import traceback
import subprocess
import argparse


def export_coreml_standalone(model, kwargs, output_dir):
    output_dir = f"{output_dir}/coreml_standalone"
    os.makedirs(output_dir, exist_ok=True)
    assert "recipe_kwargs" not in kwargs
    try:
        if kwargs.get("is_optimum", True):
            main_export(
                model_name_or_path=kwargs["model_name_or_path"],
                task=kwargs["task"],
                output_dir=output_dir,
                recipe="coreml_standalone",
            )
        else:
            ep = get_model_exporter(model)()
            lower_with_coreml(ep, filename=f"{output_dir}/model.mlpackage")
    except Exception as e:
        log_path = os.path.join(output_dir, "coreml_standalone_exception.txt")
        with open(log_path, "w") as f:
            f.write("Exception:\n")
            f.write(str(e) + "\n\n")
            f.write("Stack trace:\n")
            f.write(traceback.format_exc())

def export_coreml_et(model, kwargs, output_dir, extract_coreml_model_script = None):
    output_dir = f"{output_dir}/coreml"
    os.makedirs(output_dir, exist_ok=True)
    try:
        if kwargs.get("is_optimum", True):
            main_export(
                model_name_or_path=kwargs["model_name_or_path"],
                task=kwargs["task"],
                output_dir=output_dir,
                recipe="coreml",
                **{"recipe_kwargs": kwargs.get("recipe_kwargs", {})}
            )
        else:
            ep = get_model_exporter(model)()
            lower_with_et(ep, filename=f"{output_dir}/model.pte")
    except Exception as e:
        log_path = os.path.join(output_dir, "coreml_executorch_exception.txt")
        with open(log_path, "w") as f:
            f.write("Exception:\n")
            f.write(str(e) + "\n\n")
            f.write("Stack trace:\n")
            f.write(traceback.format_exc())
    

    model_path = f"{output_dir}/model.pte"
    if extract_coreml_model_script is not None and os.path.exists(model_path):
        subprocess.run([
            "python",
            extract_coreml_model_script,
            "-m",
            model_path
        ], cwd=output_dir)

def export_xnnpack_et(model, kwargs, output_dir):
    output_dir = f"{output_dir}/xnnpack"
    os.makedirs(output_dir, exist_ok=True)
    assert kwargs.get("is_optimum", True)
    assert "recipe_kwargs" not in kwargs
    try:
        main_export(
            model_name_or_path=kwargs["model_name_or_path"],
            task=kwargs["task"],
            output_dir=output_dir,
            recipe="xnnpack",
        )
    except Exception as e:
        log_path = os.path.join(output_dir, "xnnpack_executorch_exception.txt")
        with open(log_path, "w") as f:
            f.write("Exception:\n")
            f.write(str(e) + "\n\n")
            f.write("Stack trace:\n")
            f.write(traceback.format_exc())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process export options.")
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to the output directory"
    )
    parser.add_argument(
        "--et_repo_dir",
        type=str,
        required=False,
        default=None,
    )
    args = parser.parse_args()

    output_dir_base = args.output_dir
    extract_coreml_model_script = None
    if args.et_repo_dir is not None:
        extract_coreml_model_script = f"{args.et_repo_dir}/examples/apple/coreml/scripts/extract_coreml_models.py"

    for model, kwargs in et_optimum_ga_models.items():
        output_dir = f"{output_dir_base}/{model}"
        only_coreml_export = ("recipe_kwargs" in kwargs)
        export_coreml_et(model, kwargs, output_dir, extract_coreml_model_script=extract_coreml_model_script)
        if not only_coreml_export:
            export_coreml_standalone(model, kwargs, output_dir)
            if kwargs.get("is_optimum", True):
                export_xnnpack_et(model, kwargs, output_dir)





    