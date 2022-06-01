import yaml
import sys
import argparse
import shutil
import os

import behavior_benchmarks.evaluation.evaluation as evaluation
import behavior_benchmarks.utils.experiment_setup as experiment_setup
import behavior_benchmarks.training.train_model as train_model

def main(config):
  expanded_config = experiment_setup.experiment_setup(config)
  model = train_model.train_model(expanded_config)
  
#   import os
#   import pickle
  
#   fp = os.path.join(config['output_parent_dir'], config['experiment_name'], 'config.yaml')
#   with open(fp) as file:
#     expanded_config = yaml.load(file, Loader=yaml.FullLoader)
    
#   if not os.path.exists(expanded_config['temp_dir']):
#     os.makedirs(expanded_config['temp_dir'])
  
#   fp = os.path.join(config['output_parent_dir'], config['experiment_name'], 'final_model', 'final_model.pickle')
#   with open(fp, "rb") as input_file:
#     model = pickle.load(input_file)
    
#   from behavior_benchmarks.models.wicc import wicc
#   model.predict = predict.__get__(model, wicc)
  
  evaluation.generate_predictions(model, expanded_config)
  evaluation.generate_evaluations(expanded_config)
  
  # Clean up
  if os.path.exists(expanded_config['temp_dir']):
    shutil.rmtree(expanded_config['temp_dir'])
  
if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--config', type=str, required=True)
  args = parser.parse_args()
  config_fp = args.config
  
  with open(config_fp) as file:
    config = yaml.load(file, Loader=yaml.FullLoader)
  
  main(config)
  
  
