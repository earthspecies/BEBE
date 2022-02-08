# behavior_benchmarks

Example usage: 

0. Copy formatted data from GCP to local drive (the GCP drive is `behavior_benchmarks`)
1. Edit `behavior_benchmarks/training/example_config.yaml` to suit your desires
2. `python behavior_benchmarks/training/train_model.py --config /path/to/example_config.yaml`
3. Look at outputs directory

Currently includes GMM and es-kmeans
