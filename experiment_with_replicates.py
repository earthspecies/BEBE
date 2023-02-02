import os
import yaml
import itertools
import numpy as np
from pathlib import Path
import sys
import BEBE.evaluation.evaluation as evaluation
from tqdm import tqdm
import argparse
from plumbum import local, FG

def main(args):
  final_experiment_dir_parent = args.experiment_dir_parent
  n_experiments = args.n_replicates
  final_experiment_dir = os.path.join(final_experiment_dir_parent, args.experiment_name)
  if not os.path.exists(final_experiment_dir):
      os.makedirs(final_experiment_dir)
  
  if args.target_dir is not None:
    # If we want to choose the best experiment based on an initial hyperparameter sweep:
    model_selection_dir = Path(args.target_dir) # Directory where hyperparameter search was performed
    
    # Check if supervised or unsupervised
    config_fp = sorted(model_selection_dir.glob('**/config.yaml'))[0]
    with open(config_fp, 'r') as f:
        config = yaml.safe_load(f)
    
    # Choose hyperparameters based on best average macro f1 score (on dev or val set, for unsupervised or supervised models, respectively)
    if config['unsupervised']:
        results = model_selection_dir.glob('**/dev_eval.yaml') # unsupervised: select hyperparams based on dev set results
    else:
        results = model_selection_dir.glob('**/val_eval.yaml') # supervised: select hyperparams based on dev set results

    best_experiment = None
    best_f1 = -1
    for x in results:
        with open(x, 'r') as f:
            y = yaml.safe_load(f)
        mean_f1 = np.mean(y['individual_scores']['macro_f1s'])
        if mean_f1 > best_f1:
            best_f1 = mean_f1
            best_experiment = x.parent

    # Copy selected hyperparameters
    selected_config_fp = str(Path(best_experiment, 'config.yaml'))
  
  else:
    # Otherwise, we replicate the config file that is input
    selected_config_fp = args.target_config 
          
  with open(selected_config_fp, 'r') as f:
      config = yaml.safe_load(f)

  if args.do_not_use_val_in_train:
    config['use_val_in_train'] = False
  else:
    config['use_val_in_train'] = True # Enforce we train on all availabe data (only applies to supervised models)
  config['output_parent_dir'] = final_experiment_dir

  # Save off config files for final experiments
  config_fps = []
  for i in range(n_experiments):
      trial_name = 'trial_' + str(i)
      config['experiment_name'] = trial_name
      target_filename = trial_name + '.yaml'
      target_fp = os.path.join(final_experiment_dir, target_filename)                       
      with open(target_fp, 'w') as file:
          yaml.dump(config, file)
      config_fps.append(target_fp)

  # Run these experiments
  for config_fp in config_fps:
      local['python']['single_experiment.py',
                      f'--config={config_fp}'] & FG
      
  # Summarize final results
  results = {}
  # Test eval:
  for fold in ['test', 'dev']:
    fps = list(Path(final_experiment_dir).glob(f'**/{fold}_eval.yaml'))
    if len(fps)>0:
      results[fold] = {}
      f1s = []
      precs = []
      recs = []
      tsrs = []
      for fp in fps:
          with open(fp, 'r') as f:
              x = yaml.safe_load(f)
          n_individuals = len(x['individual_scores']['macro_f1s'])
          f1s.extend(x['individual_scores']['macro_f1s'])
          precs.extend(x['individual_scores']['macro_precisions'])
          recs.extend(x['individual_scores']['macro_recalls'])
          tsrs.extend(x['individual_scores']['time_scale_ratios'])
      #print(n_individuals)
      print(f"{fold} f1  : %1.3f (%1.3f)" % (np.mean(f1s), np.std(f1s)))
      print(f"{fold} Prec: %1.3f (%1.3f)" % (np.mean(precs), np.std(precs)))
      print(f"{fold} Rec : %1.3f (%1.3f)" % (np.mean(recs), np.std(recs)))
      print(F"{fold} TSR : %1.3f (%1.3f)" % (np.mean(tsrs), np.std(tsrs)))
      results[fold]['f1_mean'] = float(np.mean(f1s))
      results[fold]['f1_std'] = float(np.std(f1s))
      results[fold]['prec_mean'] = float(np.mean(precs))
      results[fold]['prec_std'] = float(np.std(precs))
      results[fold]['rec_mean'] = float(np.mean(recs))
      results[fold]['rec_std'] = float(np.std(recs))
      results[fold]['tsr_mean'] = float(np.mean(tsrs))
      results[fold]['tsr_std'] = float(np.std(tsrs))
  target_fp = os.path.join(final_experiment_dir, 'final_result_summary.yaml')                       
  with open(target_fp, 'w') as file:
      yaml.dump(results, file)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--target-dir', type=str, default = None, help = "path to directory containing hyperparameter selection experiments.")
  parser.add_argument('--target-config', type=str, default = None, help = "path to config that you want to replicate")
  parser.add_argument('--n_replicates', type=int, required = True, help = "how many replicates to run")
  parser.add_argument('--experiment-dir-parent', type=str, required = True, help = "parent of dir where you want to save results")
  parser.add_argument('--experiment-name', type=str, required=True, help="name of experiment")
  parser.add_argument('--do-not-use-val-in-train', action="store_true", help="Seperate out validation set (for initial hyperparameter sweep)")
  
  args = parser.parse_args()
  if args.target_config is not None:
    assert args.target_dir is None, "Cannot specify both target dir and target config"
  else:
    assert args.target_dir is not None, "If target config is not specified, must specify a directory with hyperparameter selection experiments"
  
  main(args)
