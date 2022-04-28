import os
import glob
import yaml

def expand_config(config):
  ## accepts a human-generated config dictionary
  ## and adds in a bunch of entries for access later on
  
  config['predictions_dir'] = os.path.join(config['output_dir'], 'predictions')
  config['temp_dir'] = os.path.join(config['output_dir'], 'temp')
  config['final_model_dir'] = os.path.join(config['output_dir'], "final_model")
  config['visualization_dir'] = os.path.join(config['output_dir'], "visualizations")
  
  # load metadata
  metadata_fp = os.path.join(config['dataset_dir'], 'dataset_metadata.yaml')
  with open(metadata_fp) as file:
    config['metadata'] = yaml.load(file, Loader=yaml.FullLoader)
  
  # Based on model type, decide how to save latents, predictions, and evaluation
  
  default_config_fp = os.path.join('behavior_benchmarks', 'models', 'default_configs', config['model'] + '.yaml')
  if not os.path.exists(default_config_fp):
    raise ValueError('model type not recognized, make sure there is a default config file for your model')
    
  with open(default_config_fp) as file:
    default_config = yaml.load(file, Loader=yaml.FullLoader)
  
  config['save_latents'] = default_config['save_latents']
  config['predict_and_evaluate'] = default_config['predict_and_evaluate']
  config['unsupervised'] = default_config['unsupervised']
  
  if config['save_latents']:
    config['latents_output_dir'] = os.path.join(config['output_dir'], 'latents')
    
  # Unglob data filepaths and deal with splits

  train_data_fp = []
  val_data_fp = []
  dev_data_fp = []
  test_data_fp = []
  
  data_fp_glob = os.path.join(config['dataset_dir'], 'clip_data', '*.npy')

  fps = glob.glob(data_fp_glob)
  for fp in fps:
    clip_id = fp.split('/')[-1].split('.')[0]
    if clip_id in config['metadata']['train_clip_ids']:
      train_data_fp.append(fp)
      dev_data_fp.append(fp)
    elif clip_id in config['metadata']['val_clip_ids']:
      val_data_fp.append(fp)
      dev_data_fp.append(fp)
    elif clip_id in config['metadata']['test_clip_ids']:
      test_data_fp.append(fp)
    else:
      raise ValueError("Unrecognized clip id, check dataset construction")
    
  train_data_fp.sort()
  test_data_fp.sort()
  val_data_fp.sort()
  dev_data_fp.sort()
  
  config['train_data_fp'] = train_data_fp
  config['test_data_fp'] = test_data_fp
  config['val_data_fp'] = val_data_fp
  config['dev_data_fp'] = dev_data_fp
  
  # If 'read_latents' is True, then we use the specified latent fp's as model inputs
  # The original data is still kept track of, so we can plot it and use the ground-truth labels
  if 'read_latents' in config and config['read_latents']:
    # We assume latent filenames are the same as data filenames. They are distinguished by their filepaths
    train_data_latents_fp = []
    test_data_latents_fp = []
    val_data_latents_fp = []
    dev_data_latents_fp = []
    
    for x in config['data_latents_fp_glob']:
      # Generate splits based on metadata
      fps = glob.glob(x)
      for fp in fps:
        clip_id = fp.split('/')[-1].split('.')[0]
        if clip_id in config['metadata']['train_clip_ids']:
          train_data_latents_fp.append(fp)
          dev_data_latents_fp.append(fp)
        elif clip_id in config['metadata']['val_clip_ids']:
          val_data_latents_fp.append(fp)
          dev_data_latents_fp.append(fp)
        elif clip_id in config['metadata']['test_clip_ids']:
          test_data_latents_fp.append(fp)
        else:
          raise ValueError("Unrecognized clip id")
    
    train_data_latents_fp.sort()
    test_data_latents_fp.sort()
    dev_data_latents_fp.sort()
    val_data_latents_fp.sort()
    
    config['train_data_latents_fp'] = train_data_latents_fp
    config['test_data_latents_fp'] = test_data_latents_fp
    config['val_data_latents_fp'] = val_data_latents_fp
    config['dev_data_latents_fp'] = dev_data_latents_fp
  
  else:
    config['read_latents'] = False
  
  # Set up a dictionary to keep track of file id's, the data filepaths, and (potentially) the latent filepaths:
  
  file_id_to_data_fp = {}
  file_id_to_model_input_fp = {}
  
  train_file_ids = [] # file_ids are of the form clip_id.npy, ie they are filenames
  test_file_ids = []
  val_file_ids = []
  dev_file_ids = []
  
  for fp in config['train_data_fp']:
    file_id = fp.split('/')[-1]
    file_id_to_data_fp[file_id] = fp
    train_file_ids.append(file_id)
    dev_file_ids.append(file_id)
    if not config['read_latents']:
      file_id_to_model_input_fp[file_id] = fp
  
  for fp in config['val_data_fp']:
    file_id = fp.split('/')[-1]
    file_id_to_data_fp[file_id] = fp
    val_file_ids.append(file_id)
    dev_file_ids.append(file_id)
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
    for fp in config['val_data_latents_fp']:
      file_id = fp.split('/')[-1]
      file_id_to_model_input_fp[file_id] = fp
    for fp in config['test_data_latents_fp']:
      file_id = fp.split('/')[-1]
      file_id_to_model_input_fp[file_id] = fp
  
  assert set(file_id_to_data_fp.keys()) == set(file_id_to_model_input_fp.keys()), "mismatch between specified latent filenames and data filenames"
  
  train_file_ids.sort()
  test_file_ids.sort()
  val_file_ids.sort()
  dev_file_ids.sort()
  
  config['file_id_to_data_fp'] = file_id_to_data_fp
  config['file_id_to_model_input_fp'] = file_id_to_model_input_fp
  config['train_file_ids'] = train_file_ids
  config['test_file_ids'] = test_file_ids
  config['val_file_ids'] = val_file_ids
  config['dev_file_ids'] = dev_file_ids
  
  return config

def accept_default_model_configs(config):
  # Makes sure that all the entries of the config file are properly filled in.  
  assert 'evaluation' in config
  
  if 'n_samples' not in config['evaluation']:
    if config['model'] == 'supervised_nn':
      config['evaluation']['n_samples'] = 1 ## Number of maps to sample for averaged mapping based metric. Can be time consuming.
    else:
      config['evaluation']['n_samples'] = 100
      
  ### set up model-specific config    
    
  model_type = config['model']
  model_config_name = model_type + "_config"
  
  if model_config_name not in config:
    config[model_config_name] = {}
    
  ### look up default settings
  
  default_config_fp = os.path.join('behavior_benchmarks', 'models', 'default_configs', config['model'] + '.yaml')
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
