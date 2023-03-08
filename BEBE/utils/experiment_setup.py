import os
import yaml
import glob
import warnings

def expand_config(config):
  ## accepts a human-generated config dictionary
  ## and adds in a bunch of entries for access later on
  
  config['predictions_dir'] = os.path.join(config['output_dir'], 'predictions')
  config['temp_dir'] = os.path.join(config['output_dir'], 'temp')
  config['final_model_dir'] = os.path.join(config['output_dir'], "final_model")
  config['visualization_dir'] = os.path.join(config['output_dir'], "visualizations")
  
  # Fix random seed
  if 'seed' not in config:
    warnings.warn("Random seed not specified, initializing randomly")
    config['seed'] = None
  else:
    print(f"Training with random seed {config['seed']}")
  
  # load metadata
  metadata_fp = os.path.join(config['dataset_dir'], 'dataset_metadata.yaml')
  with open(metadata_fp) as file:
    config['metadata'] = yaml.load(file, Loader=yaml.FullLoader)
  
  # Based on model type, decide how to save predictions and evaluation
  default_config_fp = os.path.join('BEBE', 'models', 'default_configs', config['model'] + '.yaml')
  if not os.path.exists(default_config_fp):
    raise ValueError('model type not recognized, make sure there is a default config file for your model')
    
  with open(default_config_fp) as file:
    default_config = yaml.load(file, Loader=yaml.FullLoader)
  
  config['unsupervised'] = default_config['unsupervised']
 
  # If model is supervised, set number of "clusters" (i.e. classes) to be the number of labels.
  # Otherwise, if number of clusters is unspecified, set it to be max(20, 4*num known labels)
  label_names = config['metadata']['label_names']
  num_labels = len(label_names)
  num_known_labels = num_labels - 1
  
  if config['unsupervised']:
    if 'num_clusters' not in config:
      config['num_clusters'] = max(20, 4 * num_known_labels) 
  else:
    config['num_clusters'] = num_labels
    
  # If data channels are unspecified, we use all available data channels.
  # By matter of convention, these are all but the last two columns in the csv files, which are reserved for individual id and behavior label
  if 'input_vars' not in config:
    config['input_vars'] = config['metadata']['clip_column_names'].copy()
    config['input_vars'].remove('individual_id')
    config['input_vars'].remove('label')
    
  # assign folds to train/val/test  
  if 'test_folds' not in config:
    warnings.warn("Test fold not specified in config file. Defaulting to development use (test fold = 0, val fold = 1)")
    test_folds = [0]
    val_folds = [1]
    train_folds = list(range(2, config['metadata']['n_folds']))
  else:
    test_folds = config['test_folds']
    if 'val_folds' in config:
      val_folds = config['val_folds']
    else:
      val_folds = []
    train_folds = list(range(config['metadata']['n_folds']))
    for fold in [test_folds, val_folds]:
      for i in fold:
        train_folds.remove(i)
  
  test_clip_ids = []
  for fold in test_folds:
    test_clip_ids.extend(config['metadata']['clip_ids_per_fold'][fold])
    
  val_clip_ids = []
  for fold in val_folds:
    val_clip_ids.extend(config['metadata']['clip_ids_per_fold'][fold])
    
  train_clip_ids = []
  for fold in train_folds:
    train_clip_ids.extend(config['metadata']['clip_ids_per_fold'][fold])
         
  # Unglob data filepaths and deal with splits
  # Simultaneously, check that all the data files are there
  
  train_data_fp = []
  val_data_fp = []
  test_data_fp = []
  
  data_fp_glob = os.path.join(config['dataset_dir'], 'clip_data', '*.csv')
  fps = glob.glob(data_fp_glob)
  for fp in fps:
    clip_id = fp.split('/')[-1].split('.')[0]
    if clip_id in test_clip_ids:
      test_data_fp.append(fp)
    elif clip_id in val_clip_ids:
      val_data_fp.append(fp)
    elif clip_id in train_clip_ids:
      train_data_fp.append(fp)
    else:
      raise ValueError("Unrecognized clip id, check dataset construction")
    
  train_data_fp.sort()
  test_data_fp.sort()
  val_data_fp.sort()

  config['train_data_fp'] = train_data_fp
  config['test_data_fp'] = test_data_fp
  config['val_data_fp'] = val_data_fp
  
  assert len([*train_data_fp, *test_data_fp, *val_data_fp]) == len(config['metadata']['clip_ids']), "mismatch between expected number of files and actual files present. check dataset creation"
  return config

def accept_default_model_configs(config):
  # Makes sure that all the entries of the config file are properly filled in.  
  ### set up model-specific config    
    
  model_type = config['model']
  model_config_name = model_type + "_config"
  
  if model_config_name not in config:
    config[model_config_name] = {}
    
  ### look up default settings
  default_config_fp = os.path.join('BEBE', 'models', 'default_configs', config['model'] + '.yaml')
  if not os.path.exists(default_config_fp):
    raise ValueError('model type not recognized, make sure there is a default config file for your model')
    
  with open(default_config_fp) as file:
    default_config = yaml.load(file, Loader=yaml.FullLoader)
    
  default_model_config = default_config['default_model_config']

  ### apply defaults if unspecified in training config file
      
  for key in default_model_config:
    if key not in config[model_config_name]:
      config[model_config_name][key] = default_model_config[key]
      
  return config

def experiment_setup(config):
  # put in default parameters if they are unspecified
  config = accept_default_model_configs(config)
  
  # create output directory
  output_dir = os.path.join(config['output_parent_dir'], config['experiment_name'])
  config['output_dir'] = output_dir
  
  # accept various defaults
  config = expand_config(config)
  
  ## save off input config
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  target_fp = os.path.join(output_dir, "config.yaml")
  with open(target_fp, 'w') as file:
    yaml.dump(config, file)

  # Set up the rest of the experiment

  if not os.path.exists(config['predictions_dir']):
    os.makedirs(config['predictions_dir'])
    
  if not os.path.exists(config['final_model_dir']):
    os.makedirs(config['final_model_dir'])
    
  if not os.path.exists(config['visualization_dir']):
    os.makedirs(config['visualization_dir'])
    
  if not os.path.exists(config['temp_dir']):
    os.makedirs(config['temp_dir'])
    
  return config
