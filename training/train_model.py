import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import glob
import yaml
import sys
import argparse
import tqdm
import shutil

sys.path.append('/home/jupyter')
import behavior_benchmarks.models as models
import behavior_benchmarks.training.evaluation as evaluation
import behavior_benchmarks.training.handle_config as handle_config
import behavior_benchmarks.visualization as bbvis

def main(config):

  ## save off input config
  
  config = handle_config.accept_default_model_configs(config)
  output_dir = os.path.join(config['output_parent_dir'], config['experiment_name'])
  
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  target_fp = os.path.join(output_dir, "config.yaml")
  with open(target_fp, 'w') as file:
    yaml.dump(config, file)

  ## modify config to be passed around in future steps

  config['output_dir'] = output_dir
  config['predictions_dir'] = os.path.join(config['output_dir'], 'predictions')
  config['temp_dir'] = os.path.join(config['output_dir'], 'temp')

  train_data_fp = []
  for x in config['train_data_fp_glob']:
    train_data_fp.extend(glob.glob(x))
  train_data_fp.sort()

  test_data_fp = []
  for x in config['test_data_fp_glob']:
    test_data_fp.extend(glob.glob(x))
  test_data_fp.sort()

  config['train_data_fp'] = train_data_fp
  config['test_data_fp'] = test_data_fp

  metadata_fp = config['metadata_fp']
  with open(metadata_fp) as file:
    config['metadata'] = yaml.load(file, Loader=yaml.FullLoader)
  
  final_model_dir = os.path.join(config['output_dir'], "final_model")
  config['final_model_dir'] = final_model_dir
  
  visualization_dir = os.path.join(config['output_dir'], "visualizations")
  config['visualization_dir'] = visualization_dir

  # Set up the rest of the experiment

  if not os.path.exists(config['predictions_dir']):
    os.makedirs(config['predictions_dir'])
    
  if not os.path.exists(config['final_model_dir']):
    os.makedirs(config['final_model_dir'])
    
  if not os.path.exists(config['visualization_dir']):
    os.makedirs(config['visualization_dir'])
    
  if not os.path.exists(config['temp_dir']):
    os.makedirs(config['temp_dir'])

  ## Instantiate model

  if config['model'] == 'gmm':
    model = models.gmm(config)
    
  elif config['model'] == 'eskmeans':
    model = models.eskmeans(config)

  else:
    raise ValueError('model type not recognized')

  # Train model

  print("Training model")
  model.fit()
  
  # Save model
  model.save()

  # Generate predictions for each file
  # Simultaneously, keep track of all predictions at once

  print("Generating predictions based on trained model")
  all_predictions = []
  all_labels = []

  for fp in tqdm.tqdm(config['train_data_fp']):
    predictions = model.predict_from_file(fp)
    predictions_fp = os.path.join(config['predictions_dir'], fp.split('/')[-1])
    np.save(predictions_fp, predictions)

    labels_idx = config['metadata']['clip_column_names'].index('label')
    labels = list(np.load(fp)[:, labels_idx])  

    all_predictions.extend(list(predictions))
    all_labels.extend(labels)

  # Evaluate

  all_labels = np.array(all_labels)
  all_predictions = np.array(all_predictions)

  eval_output_fp = os.path.join(config['output_dir'], 'train_eval.yaml')
  evaluation.perform_evaluation(all_labels, all_predictions, config, output_fp = eval_output_fp)

  # Save example figures
  
  target_fp = os.path.join(config['visualization_dir'], "confusion_matrix.png")
  bbvis.confusion_matrix(all_labels, all_predictions, config, target_fp = target_fp)

  for fp in config['train_data_fp'][:5]:
    predictions_fp = os.path.join(config['predictions_dir'], fp.split('/')[-1])
    target_filename = fp.split('/')[-1].split('.')[0] + '-track_visualization.png'
    target_fp = os.path.join(config['visualization_dir'], target_filename)
    bbvis.plot_track(fp, predictions_fp, config, target_fp = target_fp)
    
  # Clean up
  #shutil.rmtree(config['temp_dir'])
    
  print("model outputs saved to %s " % config['output_dir'])

  
if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--config', type=str, required=True)
  args = parser.parse_args()
  config_fp = args.config
  
  with open(config_fp) as file:
    config = yaml.load(file, Loader=yaml.FullLoader)
  
  main(config)