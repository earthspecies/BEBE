import os
import glob
import yaml

def expand_config(config):
  ## accepts a human-generated config dictionary
  ## and adds in a bunch of entries for access later on
  
  config['predictions_dir'] = os.path.join(config['output_dir'], 'predictions')
  config['temp_dir'] = os.path.join(config['output_dir'], 'temp')
  
  # load metadata
  metadata_fp = os.path.join(config['dataset_dir'], 'dataset_metadata.yaml')
  with open(metadata_fp) as file:
    config['metadata'] = yaml.load(file, Loader=yaml.FullLoader)
  
  # Based on model type, decide how to save latents, predictions, and evaluation
  
  if config['model'] == 'gmm':
    config['save_latents'] = False
    config['predict_and_evaluate'] = True
    
  elif config['model'] == 'kmeans':
    config['save_latents'] = False
    config['predict_and_evaluate'] = True
      
  elif config['model'] == 'whiten':
    config['save_latents'] = True
    config['predict_and_evaluate'] = False
    
  elif config['model'] == 'eskmeans':
    config['save_latents'] = False
    config['predict_and_evaluate'] = True
    
  elif config['model'] == 'vame':
    config['save_latents'] = True
    config['predict_and_evaluate'] = True # Equivalent to putting discovered latents into kmeans model
    
  elif config['model'] == 'hmm':
    config['save_latents'] = False # we treat hmm latent states as predictions
    config['predict_and_evaluate'] = True
    
  elif config['model'] == 'supervised_nn':
    config['save_latents'] = False
    config['predict_and_evaluate'] = True

  else:
    raise ValueError('model type not recognized')
  
  if config['save_latents']:
    config['latents_output_dir'] = os.path.join(config['output_dir'], 'latents')
    
  # Unglob data filepaths and deal with splits

  train_data_fp = []
  test_data_fp = []
  
  data_fp_glob = os.path.join(config['dataset_dir'], 'clip_data', '*.npy')

  fps = glob.glob(data_fp_glob)
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
  
  config['file_id_to_data_fp'] = file_id_to_data_fp
  config['file_id_to_model_input_fp'] = file_id_to_model_input_fp
  config['train_file_ids'] = train_file_ids
  config['test_file_ids'] = test_file_ids
  
  return config



def accept_default_model_configs(config):
  
  assert 'evaluation' in config
  
  if 'n_samples' not in config['evaluation']:
    config['evaluation']['n_samples'] = 100 ## Number of maps to sample for averaged mapping based metric. Can be time consuming.
    
  model_type = config['model']
  model_config_name = model_type + "_config"
  
  if model_config_name not in config:
    config[model_config_name] = {}
    
  ### look up default settings
  
  if model_type == 'gmm':
    default_model_config = {'max_iter' : 100,
                            'n_init' : 1}
      
  elif model_type == 'kmeans':      
    default_model_config = {'max_iter' : 100,
                            'n_init' : 10}
    
  elif model_type == 'eskmeans':
    default_model_config = {'landmark_hop_size' : 10,
                            'n_epochs' : 10,
                            'n_landmarks_max' : 8,
                            'embed_length' : 10,
                            'boundary_init_lambda' : 4.,
                            'max_track_len' : 10000,
                            'batch_size' : 10,
                            'time_power_term' : 1. ## 1 is standard. if between 0 and 1, it penalizes discovering short segments
                           }
    
  elif model_type == 'vame':
    default_model_config = {'batch_size' : 256,
                            'max_epochs' : 500, 
                            'beta': 1, ## Scalar multiplied by KL loss
                            'zdims': 30, ## Latent space dimensionality
                            'learning_rate' : 0.0005,
                            'time_window_sec': 1.,
                            'prediction_decoder': 1, ## Whether to predict future steps
                            'prediction_sec': 0.5, ## How much to predict after encoded window
                            'scheduler': 1,
                            'scheduler_step_size': 100,
                            'scheduler_gamma': 0.2,
                            'kmeans_lambda': 0.1, ## Scalar multiplied by kmeans loss
                            'downsizing_factor' : 4 ## Train more efficiently, account for autocorrelation
                           }
    
  elif model_type == 'whiten':
    default_model_config = {}
  
  elif model_type == 'hmm':
    default_model_config = {'time_bins' : 2500, 
                            'prior_wait_sec' : 5., 
                            'sticky_prior_strength' : 0., 
                            'N_iters' : 50, 
                            'lags' : 1
                           }
    
  elif model_type == 'supervised_nn':
    default_model_config = {}
    
  ### apply defaults if unspecified
      
  for key in default_model_config:
    if key not in config[model_config_name]:
      config[model_config_name][key] = default_model_config[key]
      
  return config
