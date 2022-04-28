import behavior_benchmarks.models as models

def train_model(config):
  ## Instantiate model

  if config['model'] == 'gmm':
    model = models.gmm(config)
    
  elif config['model'] == 'kmeans':
    model = models.kmeans(config)
    
  elif config['model'] == 'eskmeans':
    model = models.eskmeans(config)
    
  elif config['model'] == 'vame':
    model = models.vame(config)
    
  elif config['model'] == 'whiten':
    model = models.whiten(config)
    
  elif config['model'] == 'hmm':
    model = models.hmm(config)
    
  elif config['model'] == 'supervised_nn':
    model = models.supervised_nn(config)
    
  elif config['model'] == 'vq_cpc':
    model = models.vq_cpc(config)

  else:
    raise ValueError('model type not recognized')

  # Train model
  print("Training model")
  model.fit()
  
  # Save model
  model.save()
  
  return model
