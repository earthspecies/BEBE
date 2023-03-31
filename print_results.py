import yaml
from pathlib import Path
import os

experiment_dir = Path('/home/jupyter/behavior_benchmarks_experiments')

fold = 'individualized'
dataset = 'whales'
supervised = False

if fold == 'individualized':
  models = ['kmeans', 'wavelet_kmeans', 'gmm', 'hmm', 'umapper', 'vame', 'iic', 'random']
  print("individualized f1")
  for model in models:
    fn = Path(experiment_dir, f"{dataset}_{model}", "final_result_summary.yaml")
    if not os.path.exists(fn):
      print("")
      continue
    with open(fn, 'r') as f:
      results = yaml.safe_load(f)
    mu = results["train_f1_individualized_mean"]
    std = results["train_f1_individualized_std"]

    print("%1.3f (%1.3f)" % (mu, std))

else:
  for stat in ['f1', 'prec', 'rec', 'tsr']:
    print(stat)
    if supervised: 
      models = ['CNN', 'CRNN', 'rf']
    else:
      models = ['kmeans', 'wavelet_kmeans', 'gmm', 'hmm', 'umapper', 'vame', 'iic', 'random']

    for model in models:
      fn = Path(experiment_dir, f"{dataset}_{model}", "final_result_summary.yaml")
      if not os.path.exists(fn):
        print("")
        continue
      with open(fn, 'r') as f:
        results = yaml.safe_load(f)
      mu = results[fold][f"{stat}_mean"]
      std = results[fold][f"{stat}_std"]

      print("%1.3f (%1.3f)" % (mu, std))
  
