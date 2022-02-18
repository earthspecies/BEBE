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
  
  metadata_fp = config['metadata_fp']
  with open(metadata_fp) as file:
    config['metadata'] = yaml.load(file, Loader=yaml.FullLoader)
  
  if 'save_latents' in config and config['save_latents']:
    config['latents_output_dir'] = os.path.join(config['output_dir'], 'latents')
  else: 
    config['save_latents'] = False

  train_data_fp = []
  test_data_fp = []  
  
  for x in config['data_fp_glob']:
    # Generate splits based on metadata
    fps = glob.glob(x)
    for fp in fps:
      clip_id = fp.split('/')[-1].split('.')[0]
      if clip_id in config['metadata']['train_clip_ids']:
        train_data_fp.append(fp)
      else:
        test_data_fp.append(fp)
    
  train_data_fp.sort()
  test_data_fp.sort()
  
  config['train_data_fp'] = train_data_fp
  config['test_data_fp'] = test_data_fp
  
  # If 'read_latents' is True, then we use the specified latent fp's as model inputs
  # The original data is still kept track of, so we can plot it and use the ground-truth labels
  if 'read_latents' in config and config['read_latents']:
    # We assume latent filenames are the same as data filenames. They are distinguished by their filepaths
    train_data_latents_fp = []
    test_data_latents_fp = []
    
    for x in config['data_latents_fp_glob']:
      # Generate splits based on metadata
      fps = glob.glob(x)
      for fp in fps:
        clip_id = fp.split('/')[-1].split('.')[0]
        if clip_id in config['metadata']['train_clip_ids']:
          train_data_latents_fp.append(fp)
        else:
          test_data_latents_fp.append(fp)
    
    train_data_latents_fp.sort()
    test_data_latents_fp.sort()
    
    config['train_data_latents_fp'] = train_data_latents_fp
    config['test_data_latents_fp'] = test_data_latents_fp
  
  else:
    config['read_latents'] = False
  
  final_model_dir = os.path.join(config['output_dir'], "final_model")
  config['final_model_dir'] = final_model_dir
  
  visualization_dir = os.path.join(config['output_dir'], "visualizations")
  config['visualization_dir'] = visualization_dir
  
  # Set up a dictionary to keep track of file id's, the data filepaths, and (potentially) the latent filepaths:
  
  file_id_to_data_fp = {}
  file_id_to_model_input_fp = {}
  
  train_file_ids = [] # file_ids are of the form clip_id.npy, could also call them "filenames"
  test_file_ids = []
  
  for fp in config['train_data_fp']:
    file_id = fp.split('/')[-1]
    file_id_to_data_fp[file_id] = fp
    train_file_ids.append(file_id)
    if not config['read_latents']:
      file_id_to_model_input_fp[file_id] = fp
      
  for fp in config['test_data_fp']:
    file_id = fp.split('/')[-1]
    file_id_to_data_fp[file_id] = fp
    test_file_ids.append(file_id)
    if not config['read_latents']:
      file_id_to_model_input_fp[file_id] = fp
      
  if config['read_latents']:
    for fp in config['train_data_latents_fp']:
      file_id = fp.split('/')[-1]
      file_id_to_model_input_fp[file_id] = fp
    for fp in config['test_data_latents_fp']:
      file_id = fp.split('/')[-1]
      file_id_to_model_input_fp[file_id] = fp
  
  assert set(file_id_to_data_fp.keys()) == set(file_id_to_model_input_fp.keys()), "mismatch between specified latent filenames and data filenames"
  
  train_file_ids.sort()
  test_file_ids.sort()
  
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
    # kmeans model is primarily used as a way to save off whitened data as new "latents"
    # so it can be fed into eg eskmeans
    model = models.kmeans(config)
    
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
  
  choices = None # choices and probs are parameters for the mapping based metrics, we discover them using the train set on the first loop through
  probs = None
  
  for file_ids in [train_file_ids, test_file_ids]:
    print("Generating predictions based on trained model")
    all_predictions = []
    all_labels = []

    for filename in tqdm.tqdm(file_ids):
      fp = file_id_to_model_input_fp[filename]
      predictions, latents = model.predict_from_file(fp)
      predictions_fp = os.path.join(config['predictions_dir'], filename)
      np.save(predictions_fp, predictions)

      if config['save_latents']:
        latents_fp = os.path.join(config['latents_output_dir'], filename)
        np.save(latents_fp, latents)

      labels_idx = config['metadata']['clip_column_names'].index('label')
      data_fp = file_id_to_data_fp[filename]
      labels = list(np.load(data_fp)[:, labels_idx])  

      all_predictions.extend(list(predictions))
      all_labels.extend(labels)

    # Evaluate

    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    
    if file_ids == train_file_ids:
      eval_output_fp = os.path.join(config['output_dir'], 'train_eval.yaml')
      confusion_target_fp = os.path.join(config['visualization_dir'], "train_confusion_matrix.png")
    else:
      eval_output_fp = os.path.join(config['output_dir'], 'test_eval.yaml')
      confusion_target_fp = os.path.join(config['visualization_dir'], "test_confusion_matrix.png")
    eval_dict, choices, probs = evaluation.perform_evaluation(all_labels, all_predictions, config, output_fp = eval_output_fp, choices = choices, probs = probs)

  # Save example figures
    bbvis.confusion_matrix(all_labels, all_predictions, config, target_fp = confusion_target_fp)
  
  rng = np.random.default_rng(seed = 607)  # we want to plot segments chosen a bit randomly, but also consistently
  
  for file_ids in [train_file_ids, test_file_ids]:
    for filename in list(rng.choice(file_ids, 3, replace = False)):
      predictions_fp = os.path.join(config['predictions_dir'], filename)
      track_length = len(np.load(predictions_fp))
      if file_ids == train_file_ids:
        target_filename = filename.split('.')[0] + '-train-track_visualization.png'
      else:
        target_filename = filename.split('.')[0] + '-test-track_visualization.png'
      target_fp = os.path.join(config['visualization_dir'], target_filename)
      data_fp = file_id_to_data_fp[filename]
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