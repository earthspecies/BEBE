
def accept_default_model_configs(config):
  
  assert 'evaluation' in config
    
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
    
  ### apply defaults if unspecified
      
  for key in default_model_config:
    if key not in config[model_config_name]:
      config[model_config_name][key] = default_model_config[key]
      
  return config
      