# The Bio-logger Ethogram Benchmark (BEBE)

This repository contains code for model training and evaluation, presented in our forthcoming work:

> B. Hoffman, M. Cusimano, V. Baglione, D. Canestrari, D. Chevallier, D. L. DeSantis, L. Jeantet, M. Ladds, T. Maekawa, V. Mata-Silva, V. Moreno-González, A. Pagano, E. Trapote, O. Vainio, A. Vehkaoja, K. Yoda, K. Zacarian, A. Friedlaender, and C. Rutz, "A benchmark for computational analysis of animal behavior, using animal-borne tags," 2023. 

Please cite this paper if you use this code. The preprint is available at [https://arxiv.org/abs/2305.10740](https://arxiv.org/abs/2305.10740).

<img src="https://user-images.githubusercontent.com/72874445/228656286-3266a247-1935-451c-b315-6b506f7adc24.png">

For attributions of images used in this figure, see the file `ATTRIBUTIONS`.

## Install necessary Python packages:

BEBE was developed in Python 3.8, with Pytorch version 1.12.1. To set up a conda environment for BEBE, follow these steps:

```
conda create -n BEBE python=3.8 pytorch cudatoolkit=11.3 torchvision torchaudio cudnn -c pytorch -c conda-forge
conda activate BEBE
pip install -r requirements.txt
```

Please note that the version of pytorch and the cudatoolkit that you can use may vary depending on your computational resources (e.g. whether you have a GPU and what kind). Therefore, to install PyTorch you should follow the instructions on their [website](https://pytorch.org/get-started/locally/), and potentially use a [previous version](https://pytorch.org/get-started/previous-versions/) if needed. To use your GPU with our HMM module (which uses jax and dynamax), please follow the instructions [here](https://github.com/google/jax#pip-installation-gpu-cuda-installed-via-pip-easier) to install jax for GPU.

## Get data

All raw and formatted datasets for BEBE are available on our [Zenodo repository](https://zenodo.org/record/7947104).

## Run a single experiment

The directory `example_config` contains example config files. To create configs for other models included in our paper, follow the format of the example config files, and check the default configs in `BEBE/models/default_configs/` to see a list of parameters. If any parameters are not specified in your file, the `default_configs` will provide defaults values. 

To run a single experiment using these hyperparameters, edit the output directory in a config file and run `python single_experiment.py --config /path/to/CONFIG_NAME.yaml`. Note that these config files specify that training is performed using folds 1, 2, 3, and 4, and testing is performed using fold 0. After training and evaluation, results and figures can be found at the output directory specified in config file. 

To see the config files used for the experiments reported in the paper, please download the results on our [Zenodo repository](https://zenodo.org/record/7947104).

## Run multiple experiments

To replicate the experiments in the paper, run `python cross_val_experiment.py --experiment-dir-parent=/path/to/dir/where/experiment/should/go --experiment-name=EXPERIMENT NAME --dataset-dir=/path/to/formatted/dataset/ --model=MODEL TYPE --resume`

Supported model types are `CNN`, `CRNN`, `rf`, `kmeans`, `wavelet-kmeans`, `gmm`, `hmm`, `umapper`, `vame`, `iic`, and `random`.

Once these experiments have been run, final results are saved in the file `final_result_summary.yaml`. These are scores averaged across all individuals from the four test sets not used for hyperparameter selection. The per-individual scores these are based on can be found in `fold_$i/test_eval.yaml` and `fold_$i/train_eval.yaml`, listed under `individual scores`.

## Standalone evaluation

To evaluate model outputs without integrating model training into the BEBE codebase, you need to save your model predictions as `.csv` files. Predictions should be of shape `[n_samples,]`, where each entry is an integer. Each file should have the same name as a file in the original dataset. Evaluation can then be performed using the `generate_evaluations_standalone` function in `BEBE/evaluation/evaluation.py`.

## Implement a new model

1. Inherit basic structure from `BehaviorModel` which can be found in `BEBE/models/model_superclass.py`.
2. Include default model settings in a new `model_type.yaml` file, inside the directory `BEBE/models/default_configs`.
3. Include new model class inside `BEBE/training/train_model.py`.
4. To perform hyperparameter optimization as described in the paper, add your model to `BEBE/utils/hyperparameters.py`.

## Process a new dataset

Example notebooks for processing a new dataset into the format used by BEBE can be found at <https://github.com/earthspecies/BEBE-datasets/>.

## Enquiries

Please [post an issue](https://github.com/earthspecies/BEBE/issues) here on Github or [contact us](mailto:benjamin@earthspecies.org) in any of the following situations:

1. You have questions or issues with running this code.
2. You are interested in including your data in a future expansion of BEBE.
3. You test your model on BEBE and have something exciting to share.



