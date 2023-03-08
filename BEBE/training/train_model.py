def train_model(config):
  ## Instantiate model

  if config['model'] == 'gmm':
    from BEBE.models.gmm import gmm as m
    
  elif config['model'] == 'kmeans':
    from BEBE.models.kmeans import kmeans as m
    
  elif config['model'] == 'vame':
    from BEBE.models.vame import vame as m
    
  elif config['model'] == 'hmm':
    from BEBE.models.hmm import hmm as m
    
  elif config['model'] == 'CRNN':
    from BEBE.models.CRNN import CRNN as m
    
  elif config['model'] == 'umapper':
    from BEBE.models.umapper import umapper as m
    
  elif config['model'] == 'rf':
    from BEBE.models.rf import rf as m
    
  elif config['model'] == 'random':
    from BEBE.models.random import random as m
    
  elif config['model'] == 'iic':
    from BEBE.models.iic import iic as m

  else:
    raise ValueError('model type not recognized')
    
  model = m(config)

  # Train model
  print("Training model")
  model.fit()
  
  # Save model
  model.save()
  
  return model
