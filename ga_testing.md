**Repro steps for CoreML GA testing**

First create a conda environment and install ExecuTorch.

Next clone and checkout the coreml-ga-testing branch from https://github.com/metascroy/optimum-executorch/tree/coreml-ga-testing.

Install optimum-executorch by running:
```
pip install .
```
from optimum-executorch folder.

## Exporting models for GA testing

To export models for GA testing, run the following script:
```
python export_ga_models.py --output_dir "/path/to/directory/for/exported/models" --et_repo_dir "/path/to/executorch/repo/directory"
```

The above script will export various GA models with the ExecuTorch CoreML backend, the ExecuTorch XNNPACK backend, and the standalone CoreML flow.  There will be 1 folder per model, and within that a folder for coreml, coreml_standalone, and xnnpack.

If export fails, instead of a model file, you'll see a text file with a stack trace.

## Running models for GA testing
To run the GA models with pybindings, run the following:
```
python run_ga_models.py --model_dir "/path/to/directory/for/exported/models"
```

This will run the exported models 50 times each, and report the average inference time.
