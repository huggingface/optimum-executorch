First install executorch and then install this branch of optimum-executorch by running:
```
pip install .
```
from optimum-executorch folder.


To export models for GA testing, run the following command:
```
python export_ga_models.py --output_dir "/Users/scroy/Desktop/ga-model-exports" --et_repo_dir "/Users/scroy/repos/executorch"
```
This will output the exported models to "/Users/scroy/Desktop/ga-model-exports".

The models can then be run with:
```
python run_ga_models.py --model_dir "/Users/scroy/Desktop/ga-model-exports"
```

You can control what models are exported by editing ga_model_configs.py.
