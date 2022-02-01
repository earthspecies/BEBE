import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import glob
import yaml
import sys
sys.path.append('/home/jupyter')
import behavior_benchmarks.models as models
import behavior_benchmarks.training.evaluation as evaluation
import behavior_benchmarks.visualization as bbvis

def main():
  ## todo: load config and make it a dictionary

  # config = {'experiment_name' : 'gmm_test',
  #          'model' : 'gmm',
  #          'num_components' : 8,
  #          'output_parent_dir' : '/home/jupyter/behavior_benchmarks_outputs/jeantet_turtles',
  #          'input_vars' : ['AccX',
  #                          'AccY',
  #                          'AccZ',
  #                          'GyrX',
  #                          'GyrY',
  #                          'GyrZ',
  #                          'Depth'], 
  #          'metadata_fp' : '/home/jupyter/behavior_data_local/data/formatted/jeantet_turtles/dataset_metadata.yaml',
  #          'train_data_fp_glob' : ['/home/jupyter/behavior_data_local/data/formatted/jeantet_turtles/clip_data/*.npy'],
  #          'test_data_fp_glob' : ['/home/jupyter/behavior_data_local/data/formatted/jeantet_turtles/clip_data/*.npy']}

  config = {'experiment_name' : 'gmm_test',
            'model' : 'gmm',
            'num_components' : 8,
            'output_parent_dir' : '/home/jupyter/behavior_benchmarks_outputs/ladds_seals',
            'input_vars' : ['AccX',
                            'AccY',
                            'AccZ',
                            'Depth'], 
            'metadata_fp' : '/home/jupyter/behavior_data_local/data/formatted/ladds_seals/dataset_metadata.yaml',
            'train_data_fp_glob' : ['/home/jupyter/behavior_data_local/data/formatted/ladds_seals/clip_data/*.npy'],
            'test_data_fp_glob' : ['/home/jupyter/behavior_data_local/data/formatted/ladds_seals/clip_data/*.npy']}


  ## save off config

  output_dir = os.path.join(config['output_parent_dir'], config['experiment_name'])

  target_fp = os.path.join(output_dir, "config.yaml")
  with open(target_fp, 'w') as file:
    yaml.dump(config, file)

  ## modify config to be passed around

  config['output_dir'] = output_dir
  config['predictions_dir'] = os.path.join(config['output_dir'], 'predictions')

  train_data_fp = []
  test_data_fp = []
  for x in config['train_data_fp_glob']:
    train_data_fp.extend(glob.glob(x))
    train_data_fp.sort()
    ##
    #train_data_fp = train_data_fp[:2]
    ##
  for x in config['test_data_fp_glob']:
    test_data_fp.extend(glob.glob(x))
    test_data_fp.sort()
    ##
    #test_data_fp = train_data_fp[:2]
    ##

  config['train_data_fp'] = train_data_fp
  config['test_data_fp'] = test_data_fp

  metadata_fp = config['metadata_fp']
  with open(metadata_fp) as file:
    config['metadata'] = yaml.load(file, Loader=yaml.FullLoader)

  # Set up experiment

  if not os.path.exists(config['output_dir']):
    os.makedirs(config['output_dir'])

  if not os.path.exists(config['predictions_dir']):
    os.makedirs(config['predictions_dir'])

  ## Instantiate model

  if config['model'] == 'gmm':
    model = models.gmm(config)

  else:
    raise ValueError('model type not recognized')

  # Train model

  print("Training model")
  model.fit()

  # Generate predictions for each file
  # Simultaneously, keep track of all predictions at once

  print("Generating predictions based on trained model")
  all_predictions = []
  all_labels = []

  for fp in config['train_data_fp']:
    model_inputs = model.load_model_inputs(fp)
    predictions = model.predict(model_inputs)
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

  for fp in config['train_data_fp'][:5]:
    predictions_fp = os.path.join(config['predictions_dir'], fp.split('/')[-1])
    target_filename = fp.split('/')[-1].split('.')[0] + '-track_visualization.png'
    target_fp = os.path.join(config['output_dir'], target_filename)
    bbvis.plot_track(fp, predictions_fp, config, target_fp = target_fp)

  
if __name__ == "__main__":
  main()