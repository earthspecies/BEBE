import BEBE.models as models

def train_model(config):
  ## Instantiate model

  if config['model'] == 'gmm':
    model = models.gmm(config)
    
  elif config['model'] == 'kmeans':
    model = models.kmeans(config)
    
  elif config['model'] == 'vame':
    model = models.vame(config)
    
  # elif config['model'] == 'whiten':
  #   model = models.whiten(config)
    
  elif config['model'] == 'hmm':
    model = models.hmm(config)
    
  elif config['model'] == 'CRNN':
    model = models.CRNN(config)
    
  elif config['model'] == 'umapper':
    model = models.umapper(config)
    
  elif config['model'] == 'random':
    model = models.random(config)

  else:
    raise ValueError('model type not recognized')

  # Train model
  print("Training model")
  model.fit()
  
  # Save model
  model.save()
  
  return model
