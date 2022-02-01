
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
      
      
  return config
      