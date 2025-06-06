
import os
import yaml
import itertools
import numpy as np
from plumbum import local, FG
from pathlib import Path

def grid_search(model_type, 
                dataset_dir, 
                hyperparameter_selection_dir, 
                resume,
                low_data_setting,
                no_cutoff,
                nogyr,
                balance_classes
               ):
  # model_type (str) : specifies model type. For options see code
  # dataset_dir (str) : path to dataset
  # hyperparameter_selection_dir (str) : path to where hyperparameter selection experiments are held
  # resume (bool) : if true, we skip experiments which already have a saved file test_eval.yaml
  
  if not os.path.exists(hyperparameter_selection_dir):
    os.makedirs(hyperparameter_selection_dir)

  configs_list = make_configs(model_type, dataset_dir, hyperparameter_selection_dir, low_data_setting, no_cutoff, nogyr, balance_classes)
  
  for config_fp in configs_list:
    with open(config_fp, 'r') as f:
      config = yaml.safe_load(f)
    test_eval_fp = Path(hyperparameter_selection_dir, config['experiment_name'], 'test_eval.yaml')
    if resume and os.path.exists(test_eval_fp):
        continue
    try:
      local['python']['single_experiment.py',
                      f'--config={config_fp}'] & FG
    except:
      print(f"failed to execute with config {config_fp}")
  
  

def make_configs(model_type, dataset_dir, hyperparameter_selection_dir, low_data_setting, no_cutoff, nogyr, balance_classes):
  # get dataset name
  metadata_fp = Path(dataset_dir, 'dataset_metadata.yaml')
  with open(metadata_fp, 'r') as f:
    metadata = yaml.safe_load(f)
  dataset_name = metadata['dataset_name']
  
  # some model types reuse the same model code implementation
  if model_type == 'CNN' or model_type == 'RNN' or model_type == 'wavelet_RNN':
    model_type_config = 'CRNN'
  elif model_type == 'harnet_random' or model_type == 'harnet_unfrozen':
    model_type_config = 'harnet'
  elif model_type == 'wavelet_kmeans':
    model_type_config = 'kmeans'
  elif model_type == 'wavelet_rf':
    model_type_config = 'rf'
  elif model_type == 'wavelet_dt':
    model_type_config = 'dt'
  elif model_type == 'wavelet_svm':
    model_type_config = 'svm'
  else:
    model_type_config = model_type
  
  # set up hyperparameter sweep
  sweep_config = {}
  sweep_config['dataset_dir'] = [dataset_dir]
  sweep_config['model'] = [model_type_config]
  sweep_config['output_parent_dir'] = [hyperparameter_selection_dir]
  sweep_config['test_folds'] = [[0]]
  sweep_config['seed'] = [0]
  
  # Get static acc cutoff choices
  sweep_config["static_acc_cutoff_freq"] = get_static_acc_cutoff_choices(model_type, dataset_name, no_cutoff) 
  
  summary = sweep_config.copy()
  
  # get model hyperparam choices
  model_hyperparam_choices = get_model_hyperparam_choices(model_type, dataset_name)
  summary['model_config'] = model_hyperparam_choices
  
  # save summary
  target_filename = "hyperparameter_selection_summary" + '.yaml'
  target_fp = os.path.join(hyperparameter_selection_dir, target_filename)                       
  with open(target_fp, 'w') as file:
      yaml.dump(summary, file)

  # Make cartesian combinations for model_config
  sweep_model_cartesian = generate_choice_combinations(model_hyperparam_choices)

  # Incorporate into main sweep dict
  sweep_config["model_config"] = [sweep_model_cartesian[key] for key in sweep_model_cartesian]
  
  

  # Make cartesian combinations:
  sweep_config_cartesian = generate_choice_combinations(sweep_config)

  # Number so as to get experiment names
  for i in sweep_config_cartesian.keys():
      experiment_name = model_type + "_hyperparameter_selection_" + str(i)
      sweep_config_cartesian[i]['experiment_name'] = experiment_name

  # save off configs:
  config_fps = []

  for i in sorted(sweep_config_cartesian.keys()):
      config = sweep_config_cartesian[i]
      

      config['low_data_setting'] = low_data_setting
      config['balance_classes'] = balance_classes

      if nogyr:
        config['input_vars'] = get_nogyr_vars(dataset_name)
      
      target_filename = config['experiment_name'] + '.yaml'
      target_fp = os.path.join(hyperparameter_selection_dir, target_filename)
      config_fps.append(target_fp)                         
      with open(target_fp, 'w') as file:
          yaml.dump(config, file)

  return config_fps

def generate_choice_combinations(d):
    # d is a dict of lists
    # require that the keys of d are all of the same type (eg strings)
    # output a dict {i:{key:value} for i in range(N)}
    # which contains all possible combinations of values (encoded in the lists)
    # and N = product of len(d[key]), for all keys in d
    sorted_keys = sorted(d.keys())
    choices_lex = []
    num_choices = {}
    for key in sorted_keys:
        num_choices[key] = len(d[key])
        choices_lex.append(list(range(num_choices[key])))
    choices_cartesian = list(itertools.product(*choices_lex))
    configs_cartesian = {}
    for i, choices in enumerate(choices_cartesian):
        configs_cartesian[i] = {sorted_keys[j]: d[sorted_keys[j]][choices[j]] for j in range(len(sorted_keys))}
        
    return configs_cartesian
  
def get_model_hyperparam_choices(model_type, dataset_name):
  # Specify model-specific parameters
  # For each model-specific parameter, possible values for the grid search are formatted as a list
  
  if model_type == 'rf' or model_type == 'dt' or model_type == 'svm':
    # Nathan et al. 2012 features with classic ML models
    if dataset_name == 'vehkaoja_dogs':
      context_window_sec = [0.5, 1, 2, 4, 8]
    else:
      context_window_sec = [0.5, 1, 2, 4, 8, 16]
    model_hyperparam_choices = {'context_window_sec' : context_window_sec,
                                'class_weight': ['balanced'],
                                'feature_set': ['nathan2012']
                               }

    if model_type == 'rf':
      model_hyperparam_choices['n_jobs'] = [24]

  if model_type == 'wavelet_rf' or model_type == 'wavelet_dt' or model_type == 'wavelet_svm':
    # Wavelet features with classic ML models
    model_hyperparam_choices = {'context_window_sec' : [8],
                                'class_weight': ['balanced'],
                                'feature_set': ['wavelet'],
                                'wavelet_transform' : [True],
                                'whiten' : [False],
                                'morlet_w' : [5., 10., 20.],
                                'C_min' : [None],
                                'C_max' : [1., 10., 100., 1000.],
                                'n_wavelets' : [15],
                                'per_channel_normalize' : [True],
                               }

    if model_type == 'wavelet_rf':
      model_hyperparam_choices['n_jobs'] = [24]

  if model_type == 'CNN' or model_type == 'CRNN' or model_type == 'RNN' or model_type == 'wavelet_RNN':
    
    
    if model_type == 'CNN':
      gru_depth = 0
    else:
      gru_depth = 1
      
    if model_type == 'RNN' or model_type == 'wavelet_RNN':
      conv_depth = 0
      dilation = [1]
      
      # RNN and wavelet_RNN used only as ablation of harnet, use same temporal window settings
      if dataset_name == 'desantis_rattlesnakes':
        window_samples = 150
      elif dataset_name == 'ladds_seals':
        window_samples = 150
      else:
        window_samples = 900
      
    else:
      conv_depth = 2
      dilation = [1,3,5]
      
      if dataset_name == 'desantis_rattlesnakes':
        window_samples = 64
      elif dataset_name == 'ladds_seals':
        window_samples = 128
      else:
        window_samples = 2048
        
    if model_type == 'wavelet_RNN':
      wavelet_transform = True
      n_wavelets = 15
      morlet_w = [5., 10., 20.]
      C_max = [1., 10., 100., 1000.]
    else:
      wavelet_transform = False
      n_wavelets = 15
      morlet_w = [1.]
      C_max = [100.]
    
    model_hyperparam_choices = {'downsizing_factor' : [window_samples // 2],
                                'lr' : [0.01, 0.003, 0.001],
                                'weight_decay' : [0],
                                'n_epochs' : [100],
                                'hidden_size' : [64],
                                'temporal_window_samples' : [window_samples], 
                                'batch_size' : [32],
                                'conv_depth' : [conv_depth],
                                'sparse_annotations' : [True],
                                'ker_size' : [7],
                                'dilation' : dilation,
                                'gru_depth' : [gru_depth],
                                'gru_hidden_size' : [64],
                                'wavelet_transform' : [wavelet_transform],
                                'n_wavelets' : [n_wavelets],
                                'morlet_w' : morlet_w,
                                'C_min' : [None],
                                'C_max' : C_max,
                                'per_channel_normalize' : [True],
                               }
    
  if model_type == 'harnet' or model_type == 'harnet_unfrozen' or model_type == 'harnet_random':
    if dataset_name == 'ladds_seals' or dataset_name == 'desantis_rattlesnakes':
      window_samples = 150
      harnet_version = 'harnet5'
    else:
      window_samples = 900
      harnet_version = 'harnet30'
    
    if model_type == 'harnet_unfrozen' or model_type == 'harnet_random':
      freeze_encoder = False
    else:
      freeze_encoder = True
      
    if model_type == 'harnet' or model_type == 'harnet_unfrozen':
      load_pretrained_weights = True
    else:
      load_pretrained_weights = False      
      
    model_hyperparam_choices = {'downsizing_factor' : [window_samples // 2],
                                'lr' : [0.01, 0.003, .001],
                                'weight_decay' : [0],
                                'n_epochs' : [100],
                                'temporal_window_samples' : [window_samples], 
                                'batch_size' : [32],
                                'sparse_annotations' : [True],
                                'gru_hidden_size' : [64],
                                'harnet_version' : [harnet_version],
                                'freeze_encoder' : [freeze_encoder],
                                'load_pretrained_weights' : [load_pretrained_weights]
                               }

  return model_hyperparam_choices
      
def get_static_acc_cutoff_choices(model_type, dataset_name, no_cutoff):
  if model_type == "random":
    return [0]
  if model_type == "harnet":
    return [0]
  if "wavelet" in model_type:
    return [0]
  if model_type == "rf" and no_cutoff: # Use best cutoff parameter determined by full-dataset hyperparameter sweep
    print("Using static acc cutoff determined by previous hyperparameter sweep")
    if dataset_name in ["desantis_rattlesnakes"]:
      return [0]
    elif dataset_name in ["vehkaoja_dogs", "maekawa_gulls"]:
      return [0.1]
    elif dataset_name in ["ladds_seals", "pagano_bears"]:
      return [6.4]
    else:
      return [1.6]
  if model_type == "CRNN" and no_cutoff: # Use best cutoff parameter determined by full-dataset hyperparameter sweep
    print("Using static acc cutoff determined by previous hyperparameter sweep")
    if dataset_name in ["jeantet_turtles"]:
      return [0.4]
    elif dataset_name in ["ladd_seals", "pagano_bears", "friedlaender_whales"]:
      return [1.6]
    elif dataset_name in ["vehkaoja_dogs"]:
      return [6.4]
    else:
      return [0]
  if dataset_name == "desantis_rattlesnakes": # this dataset was already filtered
    return [0]
  if no_cutoff:
    return [0]
  else:
    return [0, 0.1, 0.4, 1.6, 6.4]

def get_nogyr_vars(dataset_name):
  if dataset_name in ['baglione_crows', 'desantis_rattlesnakes', 'HAR', 'maekawa_gulls']:
    return ['AccX', 'AccY', 'AccZ']
  if dataset_name in ['friedlaender_whales']:
    return ['AccX', 'AccY', 'AccZ', 'Depth', 'Speed']
  if dataset_name in ['pagano_bears']:
    return ['AccX', 'AccY', 'AccZ', 'Wetdry']
  if dataset_name in ['jeantet_turtles', 'ladds_seals']:
    return ['AccX', 'AccY', 'AccZ', 'Depth']
  if dataset_name in ['vehkaoja_dogs']:
    return ['AccX_Back', 'AccY_Back', 'AccZ_Back', 'AccX_Neck', 'AccY_Neck', 'AccZ_Neck']
  else:
    print("Nogyr dataset name not recognized, using tri-axial accelerometer channels")
    return ["AccX", "AccY", "AccZ"]
      