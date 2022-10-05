# The Bio-logger Ethogram Benchmark (BEBE)

## Problem setup and commentary

Each dataset consists of a number of csv files, which are all of the format `[n_samples, n_features + 2]`. The last two columns contain individual id and ground truth label, respectively. Label number `0` is always reserved for unknown. (This should all be handled by the existing functions; for typical use you only need to call `model_class_instance.load_model_inputs(filepath)` to read data in from a dataset file). Each dataset comes with a `metadata.yaml` file, which has column names, sample rate, etc.

The task is to predict, for each sample, the behavior label (supervised) or cluster index (unsupervised). So models should output predictions of the form `[n_samples,]`, where each entry is an integer. 

In the code, the data is split into `train`, `val`, and `test` sets. For training unsupervised models, and also for training supervised models after performing hyperparameter selection, we use both `train` and `val` sets. These two sets together are called the `dev` set. Note this differs from the terminology in the manuscript (where `val` isn't mentioned by name, and where `dev` is called the train set).

In order to evaluate model performance, we compute classification metrics using predicted behavior labels. We also compute `time scale ratio`, which is supposed to capture how over- or under-segmented the predictions are compared to the ground truth labels. For unsupervised models, it is necessary to also perform the contingency analysis step described in the manuscript.

For all performence scores, we average scores across all individuals being used for evaluation. These `individual_scores` are reported in `test_eval.yaml`, as well as `dev_eval.yaml` (for unsupervised models). To obtain the final scores we report in the manuscript, individual scores are also averaged across three training runs.

## Install necessary Python packages:

```
cat requirements.txt | sed -e '/^\s*#.*$/d' -e '/^\s*$/d' | xargs -n 1 python -m pip install
pip install -e BEBE/applications/ssm/
```

## Get data

Copy formatted data from GCP to local drive (the GCP drive is `behavior_benchmarks`). To do this, make sure your machine is linked to the earthspecies cloud storage and run `gcsfuse --implicit-dirs behavior_benchmarks MOUNTING_DIR`. Here `MOUNTING_DIR` should be a path to an empty directory.

## Run experiment

Edit `behavior_benchmarks/example_config/example_config_NAME.yaml` to suit your desires. If model-specific parameters aren't specified, the default values are used (see "Implement a new model", below)

Run `python full_experiment.py --config /path/to/example_config_NAME.yaml`

Look at outputs directory, which was specified in config file. You can also find nice pictures there.

## Run multiple experiments

There are three notebooks which are useful:

`grid_search_setup.ipynb`: Sets up config files for a grid search across specified hyperparameters.

`final_experiment_setup.ipynb`: Selects optimal hyperparameters and duplicates config files, in order to run multiple trials for the final experiment.

`get_final_results.ipynb`: Computes mean and standard deviation for scores, across multiple individuals and multiple training runs. This is used to compute the final scores for the manuscript.

## Implement new model

1. Inherit basic structure from `BehaviorModel` which can be found in `behavior_benchmarks/models/model_superclass.py`
2. Include default model settings in a new `model_type.yaml` file, inside the directory `behavior_benchmarks/models/default_configs`
3. Include new model class inside `train_model.py` and inside `behavior_benchmarks/models/__init__.py`
