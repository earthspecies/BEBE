import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import glob
import yaml
import sys
import behavior_benchmarks.models as models

## todo: wrap in a function etc
## todo: load config and make it a dictionary

config = {'experiment_name' : 'gmm_test',
          'model' : 'gmm',
          'num_components' : 8,
          'output_parent_dir' : '/home/jupyter/behavior_benchmarks_outputs/jeantet_turtles',
          'input_vars' : ['AccX',
                          'AccY',
                          'AccZ',
                          'GyrX',
                          'GyrY',
                          'GyrZ',
                          'Depth'], 
          'metadata_fp' : '/home/jupyter/behavior_data_local/data/formatted/jeantet_turtles/dataset_metadata.yaml',
          'train_data_fp_glob' : ['/home/jupyter/behavior_data_local/data/formatted/jeantet_turtles/clip_data/*.npy'],
          'test_data_fp_glob' : ['/home/jupyter/behavior_data_local/data/formatted/jeantet_turtles/clip_data/*.npy']}

##

# modify config 

config['output_dir'] = os.path.join(config['output_parent_dir'], config['experiment_name'])

train_data_fp = []
test_data_fp = []
for x in config['train_data_fp_glob']:
  train_data_fp.extend(glob.glob(x))
  ##
  train_data_fp = train_data_fp[:2]
  ##
for x in config['test_data_fp_glob']:
  test_data_fp.extend(glob.glob(x))
  ##
  test_data_fp = train_data_fp[:2]
  ##

config['train_data_fp'] = train_data_fp
config['test_data_fp'] = test_data_fp

# Get dataset metadata

metadata_fp = config['metadata_fp']
with open(metadata_fp) as file:
  metadata = yaml.load(file, Loader=yaml.FullLoader)

# Set up experiment
                  
if not os.path.exists(config['output_dir']):
  os.makedirs(config['output_dir'])
  
## Instantiate model

if config['model'] == 'gmm':
  model = models.gmm(config)
  
else:
  raise ValueError('model type not recognized')

# Train model
model.fit()

# Generate predictions for each file

all_predictions = []
all_labels = []

labels_idx = metadata['clip_column_names'].index('label')

for fp in config['train_data_fp'][:2]:
  predictions = list(model.predict(fp))
  labels = list(np.load(fp)[:, labels_idx])
  all_predictions.append(predictions)
  all_labels.append(labels)
  
