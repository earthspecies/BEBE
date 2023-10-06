# example usage:
# python cross_val_experiment.py --experiment-dir-parent=/home/jupyter/behavior_benchmarks_experiments --experiment-name=harnet_turtles --dataset-dir=/home/jupyter/behavior_data_local/jeantet_turtles/ --model=harnet --resume

import os
import yaml
import numpy as np
import argparse
from plumbum import local, FG
from BEBE.utils.hyperparameters import grid_search
from BEBE.evaluation.cross_val_evaluation import cross_val_evaluation
from glob import glob

def main(args):
  experiment_dir_parent = args.experiment_dir_parent
  
  # hyperparameter selection on fold 0
  hyperparameter_selection_dir = os.path.join(experiment_dir_parent, f"{args.experiment_name}_hyperparameter_selection")
  if not os.path.exists(hyperparameter_selection_dir):
      os.makedirs(hyperparameter_selection_dir)
  
  grid_search(args.model, args.dataset_dir, str(hyperparameter_selection_dir), args.resume, args.low_data_setting, args.no_cutoff, args.nogyr)
  
  # choose hyperparameters based on f1 score
  best_experiment = None
  best_f1 = -1
  for x in glob(os.path.join(hyperparameter_selection_dir, '**/test_eval.yaml')):
      with open(x, 'r') as f:
          y = yaml.safe_load(f)
      mean_f1 = np.mean(y['individual_scores']['macro_f1s'])
      if mean_f1 > best_f1:
          best_f1 = mean_f1
          best_experiment = os.path.dirname(x)
  
  # Copy selected hyperparameters
  selected_config_fp = os.path.join(best_experiment, 'config.yaml')
  
  with open(selected_config_fp, 'r') as f:
      config = yaml.safe_load(f)
      
  # Set up final experiments
  final_experiment_dir = os.path.join(experiment_dir_parent, args.experiment_name)    
  if not os.path.exists(final_experiment_dir):
      os.makedirs(final_experiment_dir)
      
  config['output_parent_dir'] = final_experiment_dir
  folds = list(range(1,5))
  
  for fold in folds:
      trial_name = f"fold_{fold}"
      config['experiment_name'] = trial_name
      target_filename = trial_name + '.yaml'
      config['test_folds'] = [fold]
      config['seed'] = fold
      config_fp = os.path.join(final_experiment_dir, f"{trial_name}.yaml")                       
      with open(config_fp, 'w') as file:
          yaml.dump(config, file)

  # Run these experiments
  for fold in folds:
      trial_name = f"fold_{fold}"
      config_fp = os.path.join(final_experiment_dir, f"{trial_name}.yaml")
      test_eval_fp = os.path.join(final_experiment_dir, trial_name, 'test_eval.yaml')
      if args.resume and os.path.exists(test_eval_fp):
        continue
      local['python']['single_experiment.py',
                      f'--config={config_fp}'] & FG
      
  # Summarize final results
  cross_val_evaluation(final_experiment_dir, config['metadata'])

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--experiment-dir-parent', type=str, required = True, help = "parent of dir where you want to save results")
  parser.add_argument('--experiment-name', type=str, required=True, help="name of experiment")
  parser.add_argument('--dataset-dir', type=str, required=True, help="path to dir where formatted dataset is stored")
  parser.add_argument('--model', type=str, required=True, help="name of model type being tested", choices = ['rf', 'CNN', 'CRNN', 'RNN', 'kmeans', 'wavelet_kmeans', 'gmm', 'hmm', 'umapper', 'vame', 'iic', 'random', 'harnet', 'harnet_unfrozen', 'harnet_random'])
  parser.add_argument('--resume', action='store_true', help="skip experiments if test_eval file already exists")
  parser.add_argument('--low-data-setting', action='store_true', help="use only one fold for training, to simulate setting with low data")
  parser.add_argument('--no-cutoff', action='store_true', help="skip separating static and dynamic acc using low pass filter")
  parser.add_argument('--nogyr', action='store_true', help="omit gyr channels")
  args = parser.parse_args()
  main(args)