import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import yaml
import sys
import argparse
import tqdm
import shutil

import behavior_benchmarks.models as models
import behavior_benchmarks.training.evaluation as evaluation
import behavior_benchmarks.training.handle_config as handle_config
import behavior_benchmarks.visualization as bbvis

def main(config):
  
  # put in default parameters if they are unspecified
  config = handle_config.accept_default_model_configs(config)
  
  # create output directory
  output_dir = os.path.join(config['output_parent_dir'], config['experiment_name'])
  
  ## save off input config
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  target_fp = os.path.join(output_dir, "config.yaml")
  with open(target_fp, 'w') as file:
    yaml.dump(config, file)

  ## modify config to be passed around in future steps
  config['output_dir'] = output_dir
  config = handle_config.expand_config(config)

  # Set up the rest of the experiment

  if not os.path.exists(config['predictions_dir']):
    os.makedirs(config['predictions_dir'])
    
  if not os.path.exists(config['final_model_dir']):
    os.makedirs(config['final_model_dir'])
    
  if not os.path.exists(config['visualization_dir']):
    os.makedirs(config['visualization_dir'])
    
  if not os.path.exists(config['temp_dir']):
    os.makedirs(config['temp_dir'])
    
  if config['save_latents'] and not os.path.exists(config['latents_output_dir']):
    os.makedirs(config['latents_output_dir'])

  ## Instantiate model

  if config['model'] == 'gmm':
    model = models.gmm(config)
    
  elif config['model'] == 'kmeans':
    model = models.kmeans(config)
    
  elif config['model'] == 'eskmeans':
    model = models.eskmeans(config)
    
  elif config['model'] == 'vame':
    model = models.vame(config)
    
  elif config['model'] == 'whiten':
    model = models.whiten(config)
    
  elif config['model'] == 'hmm':
    model = models.hmm(config)

  else:
    raise ValueError('model type not recognized')

  # Train model
  print("Training model")
  model.fit()
  
  # Save model
  
  model.save()

  # Generate predictions for each file
  # Simultaneously, keep track of all predictions at once
  
  choices = None # choices and probs are parameters for the mapping based metrics, we discover them using the train set on the first loop through
  probs = None
  
  for file_ids in [config['train_file_ids'], config['test_file_ids']]:
    print("Generating predictions & latents based on trained model")
    all_predictions = []
    all_labels = []

    for filename in tqdm.tqdm(file_ids):
      fp = config['file_id_to_model_input_fp'][filename]
      predictions, latents = model.predict_from_file(fp)
      
      if config['predict_and_evaluate']:
        predictions_fp = os.path.join(config['predictions_dir'], filename)
        np.save(predictions_fp, predictions)

      if config['save_latents']:
        latents_fp = os.path.join(config['latents_output_dir'], filename)
        np.save(latents_fp, latents)

      labels_idx = config['metadata']['clip_column_names'].index('label')
      data_fp = config['file_id_to_data_fp'][filename]
      labels = list(np.load(data_fp)[:, labels_idx])  

      all_predictions.extend(list(predictions))
      all_labels.extend(labels)

    # Evaluate

    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    
    if file_ids == config['train_file_ids']:
      eval_output_fp = os.path.join(config['output_dir'], 'train_eval.yaml')
      confusion_target_fp = os.path.join(config['visualization_dir'], "train_confusion_matrix.png")
    else:
      eval_output_fp = os.path.join(config['output_dir'], 'test_eval.yaml')
      confusion_target_fp = os.path.join(config['visualization_dir'], "test_confusion_matrix.png")
      
    if config['predict_and_evaluate']:
      eval_dict, choices, probs = evaluation.perform_evaluation(all_labels, all_predictions, config, output_fp = eval_output_fp, choices = choices, probs = probs, n_samples = config['evaluation']['n_samples'])

      # Save confusion matrix
      bbvis.confusion_matrix(all_labels, all_predictions, config, target_fp = confusion_target_fp)
  
  # Save example figures
  
  if config['predict_and_evaluate']:
    rng = np.random.default_rng(seed = 607)  # we want to plot segments chosen a bit randomly, but also consistently

    for file_ids in [config['train_file_ids'], config['test_file_ids']]:
      for filename in list(rng.choice(file_ids, 3, replace = False)):
        predictions_fp = os.path.join(config['predictions_dir'], filename)
        track_length = len(np.load(predictions_fp))
        if file_ids == config['train_file_ids']:
          target_filename = filename.split('.')[0] + '-train-track_visualization.png'
        else:
          target_filename = filename.split('.')[0] + '-test-track_visualization.png'
        target_fp = os.path.join(config['visualization_dir'], target_filename)
        data_fp = config['file_id_to_data_fp'][filename]
        if track_length <= 20000:
          bbvis.plot_track(data_fp, predictions_fp, config, eval_dict, target_fp = target_fp, start_sample = max(0, track_length - 20000), end_sample = track_length)
        else:
          start_sample = rng.integers(0, high = track_length - 20000)
          bbvis.plot_track(data_fp, predictions_fp, config, eval_dict, target_fp = target_fp, start_sample = start_sample, end_sample = start_sample + 20000)

  # Clean up
  
  shutil.rmtree(config['temp_dir'])
  print("model outputs saved to %s " % config['output_dir'])

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--config', type=str, required=True)
  args = parser.parse_args()
  config_fp = args.config
  
  with open(config_fp) as file:
    config = yaml.load(file, Loader=yaml.FullLoader)
  
  main(config)