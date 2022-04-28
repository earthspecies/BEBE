import yaml
import sys
import argparse
import shutil

import behavior_benchmarks.evaluation.evaluation as evaluation
import behavior_benchmarks.utils.experiment_setup as experiment_setup
import behavior_benchmarks.training.train_model as train_model


def main(config):
  expanded_config = experiment_setup.experiment_setup(config)
  model = train_model.train_model(expanded_config)
  evaluation.generate_predictions_and_evaluate(model, expanded_config)
  
  # Clean up
  shutil.rmtree(config['temp_dir'])
  
if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--config', type=str, required=True)
  args = parser.parse_args()
  config_fp = args.config
  
  with open(config_fp) as file:
    config = yaml.load(file, Loader=yaml.FullLoader)
  
  main(config)