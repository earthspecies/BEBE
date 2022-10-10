# from BEBE.models.kmeans import kmeans
# from BEBE.models.vame import vame
# from BEBE.models.hmm import hmm
# from BEBE.models.umapper import umapper
# from BEBE.models.random import random
# from BEBE.models.CRNN import CRNN
# from BEBE.models.rf import rf

def train_model(config):
  ## Instantiate model

  if config['model'] == 'gmm':
    from BEBE.models.gmm import gmm as m
    
  elif config['model'] == 'kmeans':
    from BEBE.models.kmeans import kmeans as m
    #model = models.kmeans(config)
    
  elif config['model'] == 'vame':
    from BEBE.models.vame import vame as m
    #model = models.vame(config)
    
  elif config['model'] == 'hmm':
    from BEBE.models.hmm import hmm as m
    #model = models.hmm(config)
    
  elif config['model'] == 'CRNN':
    from BEBE.models.CRNN import CRNN as m
    #model = models.CRNN(config)
    
  elif config['model'] == 'umapper':
    from BEBE.models.umapper import umapper as m
    #model = models.umapper(config)
    
  elif config['model'] == 'rf':
    from BEBE.models.rf import rf as m
    #model = models.rf(config)
    
  elif config['model'] == 'random':
    from BEBE.models.random import random as m
    #model = models.random(config)
    
  elif config['model'] == 'wicc':
    from BEBE.models.wicc import wicc as m
    #model = wicc(config)
    
  elif config['model'] == 'sashimi':
    from BEBE.models.sashimi import sashimi as m

  else:
    raise ValueError('model type not recognized')
    
  model = m(config)

  # Train model
  print("Training model")
  model.fit()
  
  # Save model
  model.save()
  
  return model
