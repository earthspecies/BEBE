
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
                            'kmeans_lambda': 0.1 ## Scalar multiplied by kmeans loss
                           }
    
  ### apply defaults if unspecified
      
  for key in default_model_config:
    if key not in config[model_config_name]:
      config[model_config_name][key] = default_model_config[key]
      
  return config
      