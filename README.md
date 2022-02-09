# behavior_benchmarks

Currently includes GMM and es-kmeans

## Install necessary Python packages:

```
pip install -r requirements.txt
pip install -e .
```

## Get data

Copy formatted data from GCP to local drive (the GCP drive is `behavior_benchmarks`)

## Set up experiment

Edit `behavior_benchmarks/example_config/example_config_NAME.yaml` to suit your desires

## Run experiment

`python train_model.py --config /path/to/example_config_NAME.yaml`

## Evaluation

Look at outputs directory, which was specified in config file
