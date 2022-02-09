
def accept_default_model_configs(config):
  model_type = config['model']
  
  if model_type == 'gmm':
    if 'gmm_config' not in config:
      config['gmm_config'] = {}
    
    if 'num_components' not in config['gmm_config']:
      config['gmm_config']['num_components'] = 10
      
    if 'max_iter' not in config['gmm_config']:
      config['gmm_config']['max_iter'] = 100
      
    if 'n_init' not in config['gmm_config']:
      config['gmm_config']['n_init'] = 1
  
  if model_type == 'eskmeans':
    if 'eskmeans_config' not in config:
      config['eskmeans_config'] = {}
      
    if 'landmark_hop_size' not in config['eskmeans_config']:
      config['eskmeans_config']['landmark_hop_size'] = 10
      
    if 'n_epochs' not in config['eskmeans_config']:
      config['eskmeans_config']['n_epochs'] = 10
      
    if 'n_landmarks_max' not in config['eskmeans_config']:
      config['eskmeans_config']['n_landmarks_max'] = 8
      
    if 'embed_length' not in config['eskmeans_config']:
      config['eskmeans_config']['embed_length'] = 10
      
    if 'n_clusters' not in config['eskmeans_config']:
      config['eskmeans_config']['n_clusters'] = 8
      
    if 'boundary_init_lambda' not in config['eskmeans_config']:
      config['eskmeans_config']['boundary_init_lambda'] = 4.
      
    if 'max_track_len' not in config['eskmeans_config']:
      config['eskmeans_config']['max_track_len'] = 10000
      
    if 'batch_size' not in config['eskmeans_config']:
      config['eskmeans_config']['batch_size'] = 10
      
    if 'time_power_term' not in config['eskmeans_config']:
      config['eskmeans_config']['time_power_term'] = 1. ## 1 is standard. if between 0 and 1, it penalizes discovering short segments
      
  return config
      