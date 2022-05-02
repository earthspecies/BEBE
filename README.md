# behavior_benchmarks

Currently includes: GMM, HMM, es-kmeans, VAME, VQ-CPC, k-means, whitening transform, and a supervised LSTM model

## Problem setup and commentary

Each dataset consists of a number of numpy arrays, which are all of the format `[n_samples, n_features + 2]`. The last two columns contain individual id and ground truth label, respectively. Label number `0` is always reserved for unknown. (This should all be handled by the existing functions; you only need to call `model_class_instance.load_model_inputs(filepath)` to read data in from a dataset file. Models can be set to only save off discovered latent vectors, and also can be set to read in another model's saved-off latents.

The task is to predict, for each sample, a cluster label. So models should output predictions of the form `[n_samples,]`, where each entry is an integer cluster label.

We specify the number of discoverable cluster labels <img src="https://render.githubusercontent.com/render/math?math=M"> in the model setup. Having a static number of cluster labels makes it easier to compare different models. I have been using the number of cluster labels to be 4 times the number of ground truth class labels <img src="https://render.githubusercontent.com/render/math?math=N">. That is, <img src="https://render.githubusercontent.com/render/math?math=M = 4N">. Ground truth classes can be found in each dataset's metadata file.

For unsupervised models, I have been lumping the `train` and `val` splits into `dev`, which I use for training. Model selection is performed in a supervised manner, using the entire `dev` split. For our proposed best model, it will be desireable to have unsupervised model selection, but I will consider this problem later on. For the supervised model, I use `train`, `val`, `test` splits in the typical way. Individuals only appear in one split.

In order to evaluate model performance, we assume that each cluster is a subset of a single ground truth behavior class. That is, there is an unknown many-to-one function <img src="https://render.githubusercontent.com/render/math?math=F\colon \{1,\dots, M\} \to \{1,\dots, N\}"> which assigns each cluster to its true behavior label. We estimate <img src="https://render.githubusercontent.com/render/math?math=F"> by setting <img src="https://render.githubusercontent.com/render/math?math=\hat{F}(i) = \text{argmax}_j |c_i \cap l_j|">, where <img src="https://render.githubusercontent.com/render/math?math=c_i"> denotes the set of samples assigned to the <img src="https://render.githubusercontent.com/render/math?math=i^{th}"> cluster, and <img src="https://render.githubusercontent.com/render/math?math=l_j"> denotes the set of samples assigned to the <img src="https://render.githubusercontent.com/render/math?math=j^{th}"> label. We compute classification metrics using labels predicted by <img src="https://render.githubusercontent.com/render/math?math=\hat{F}">. This is reported as `macro MAP f1`, etc.

We also save off two `consistency` plots, which are meant to be diagnostic tools to see if a model is able to disentangle behavior from individual identity. Let <img src="https://render.githubusercontent.com/render/math?math=\hat{F}_k"> be an estimate of <img src="https://render.githubusercontent.com/render/math?math=F"> as before, except now it is only based upon samples taken from individual <img src="https://render.githubusercontent.com/render/math?math=k">. The `consistency` plot measures how <img src="https://render.githubusercontent.com/render/math?math=\hat{F}_k"> varies with <img src="https://render.githubusercontent.com/render/math?math=k">, by plotting individual-wise `f1` scores computed using <img src="https://render.githubusercontent.com/render/math?math=\hat{F}_k"> in comparison with overall `f1` score computed using <img src="https://render.githubusercontent.com/render/math?math=\hat{F}">.

We also report cluster homogeneity, which is an information theoretic metric.

There are some legacy metrics, such as `average f1` and `R score` which should be ignored. I can explain why in person.

## Install necessary Python packages:

```
pip install -r requirements.txt
pip install -e .
```

## Get data

Copy formatted data from GCP to local drive (the GCP drive is `behavior_benchmarks`). For now, we will focus on the following datasets:
- humans (HAR)
- dogs
- turtles 
HAR is very easy for supervised models but still challenging for unsupervised models. The seals dataset is weird, but small enough to rapidly test if code runs. The polar bears dataset is huge.

## Set up experiment

Edit `behavior_benchmarks/example_config/example_config_NAME.yaml` to suit your desires

The notebook `experiment_setup.ipynb` is useful for setting up grid searches, but it's not meant to be part of the final package

## Run experiment

`python full_experiment.py --config /path/to/example_config_NAME.yaml`

## Evaluation

Look at outputs directory, which was specified in config file. You can also find nice pictures there.

## Implement new model

1. Inherit basic structure from `BehaviorModel` which can be found in `behavior_benchmarks/models/model_superclass.py`
2. Include default model settings in a new `model_type.yaml` file, inside the directory `behavior_benchmarks/models/default_configs`
3. Include new model class inside `train_model.py` and inside `behavior_benchmarks/models/__init__.py`

## Known shortcomings

- I haven't worried enough about model saving and loading, including checkpointing.
- There is code that is less beautiful than desired
