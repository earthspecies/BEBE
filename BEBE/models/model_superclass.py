import yaml
import numpy as np
import pickle
import os

class BehaviorModel():
  def __init__(self, config):
    self.config = config
    self.read_latents = config['read_latents']
    
    model_type = self.config['model']
    self.model_config = config[model_type + '_config']
    self.metadata = config['metadata']
      
    cols_included_bool = [x in self.config['input_vars'] for x in self.metadata['clip_column_names']] 
    self.cols_included = [i for i, x in enumerate(cols_included_bool) if x]
    
  def load_model_inputs(self, filepath, read_latents = False):
    if read_latents:
      return np.genfromtxt(filepath, delimiter = ',')
      #return np.load(filepath)
    else:
      return np.genfromtxt(filepath, delimiter = ',')[:, self.cols_included]
      #return np.load(filepath)[:, self.cols_included]
    
  def fit(self):
    ##
    # to implement in subclass
    ##
    pass
    
  def save(self):
    ##
    # possibly to implement in subclass
    ##
    target_fp = os.path.join(self.config['final_model_dir'], "final_model.pickle")
    with open(target_fp, 'wb') as f:
      pickle.dump(self, f)
  
  def predict(self, data):
    ##
    # to implement in subclass
    ##
    predictions = None
    latents = None
    
    # Returns this signature:
    return predictions, latents
  
  def predict_from_file(self, fp):
    inputs = self.load_model_inputs(fp, read_latents = self.read_latents)
    predictions, latents = self.predict(inputs)
    return predictions, latents
