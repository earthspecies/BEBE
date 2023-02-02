# The Bio-logger Ethogram Benchmark (BEBE)

## Problem setup and commentary

Each dataset consists of a number of csv files, which are all of the format `[n_samples, n_features + 2]`. The last two columns contain individual id and ground truth label, respectively. Label number `0` is always reserved for unknown. Each dataset comes with a `metadata.yaml` file, which has column names, sample rate, etc.

The task is to predict, for each sample, the behavior label (supervised) or cluster index (unsupervised). Models output predictions of the form `[n_samples,]`, where each entry is an integer. 

In the code, the data is split into `train`, `val`, and `test` sets. For training unsupervised models, and also for training supervised models after performing hyperparameter selection, we use both `train` and `val` sets. These two sets together are called the `dev` set. Note this differs from the terminology in the article (where `val` isn't mentioned by name, and where `dev` is called the train set).

In order to evaluate model performance, we compute classification metrics using predicted behavior labels. We also compute `time scale ratio`, which is supposed to capture how over- or under-segmented the predictions are compared to the ground truth labels. For unsupervised models, it is necessary to also perform the contingency analysis step described in the manuscript.

For all performence scores, we average scores across all individuals being used for evaluation. These `individual_scores` are reported in `test_eval.yaml`, as well as `dev_eval.yaml` (for unsupervised models). To obtain the final scores we report in the manuscript, individual scores are also averaged across three training runs. This is performed by `experiment_with_replicates.py`.

## Install necessary Python packages:

```
conda create -n BEBE python=3.8 pytorch cudatoolkit=11.3 torchvision torchaudio cudnn -c pytorch -c conda-forge
conda activate BEBE
pip install -r requirements.txt
pip install -e BEBE/applications/ssm/
```

## Get data

Copy formatted data from GCP to local drive (the GCP drive is `behavior_benchmarks`). I use `gsutil -m cp -r gs://behavior_benchmarks/formatted/DATASET_NAME /home/jupyter/behavior_data_local/data/`.

## Run experiment

Edit `behavior_benchmarks/example_config/example_config_NAME.yaml` to suit your desires. If model-specific parameters aren't specified, the default values are used (see "Implement a new model", below)

Run `python single_experiment.py --config /path/to/example_config_NAME.yaml`

Look at outputs directory, which was specified in config file. You can also find figures there.

## Run multiple experiments

For selecting hyperparameters, the notebook `grid_search_setup.ipynb` sets up config files for a grid search across specified hyperparameters.

Once these experiments have been run, use `experiment_with_replicates.py` to choose the best set of hyperparameters, run experiment replicates, and save final evaluation metrics.

## How we did it

We performed an initial hyperparameter sweep using `grid_search_setup.ipynb`, for the hyperparameters described in the paper. Then, we used `experiment_with_replicates.py` with to select hyperperameters from our hyperparameter search directory (`target_dir`), and perform `n_replicates=3` replicates.

## Standalone evaluation

To evaluate model outputs without integrating model training into the BEBE codebase, you need to save your model predictions as `.csv` files. Predictions should be of shape `[n_samples,]`, where each entry is an integer. Each file should have the same name as a file in the original dataset. Evaluation can then be performed using the `generate_evaluations_standalone` function in `BEBE/evaluation/evaluation.py`.

## Implement new model

1. Inherit basic structure from `BehaviorModel` which can be found in `behavior_benchmarks/models/model_superclass.py`
2. Include default model settings in a new `model_type.yaml` file, inside the directory `behavior_benchmarks/models/default_configs`
3. Include new model class inside `train_model.py` and inside `behavior_benchmarks/models/__init__.py`
