from sklearn.mixture import GaussianMixture
from BEBE.models.model_superclass import BehaviorModel
import yaml
import numpy as np
import pickle
import os

class gmm(BehaviorModel):
  def __init__(self, config):
    super(gmm, self).__init__(config)
    # self.config = config
    # self.read_latents = config['read_latents']
    # self.model_config = config['gmm_config']
    self.model = GaussianMixture(n_components = self.config['num_clusters'], verbose = 2, max_iter = self.model_config['max_iter'], n_init = self.model_config['n_init'])
    # self.metadata = config['metadata']
      
#     cols_included_bool = [x in self.config['input_vars'] for x in self.metadata['clip_column_names']] 
#     self.cols_included = [i for i, x in enumerate(cols_included_bool) if x]
  
  # def load_model_inputs(self, filepath, read_latents = False):
  #   if read_latents:
  #     return np.load(filepath)
  #   else:
  #     return np.load(filepath)[:, self.cols_included]
    
  def fit(self):
    ## get data. assume stored in memory for now
    if self.read_latents:
      dev_fps = self.config['dev_data_latents_fp']
    else:
      dev_fps = self.config['dev_data_fp']
    
    dev_data = [self.load_model_inputs(fp, read_latents = self.read_latents) for fp in dev_fps]
    dev_data = np.concatenate(dev_data, axis = 0)

    self.model.fit(dev_data)
    
  def save(self):
    target_fp = os.path.join(self.config['final_model_dir'], "final_model.pickle")
    with open(target_fp, 'wb') as f:
      pickle.dump(self, f)
  
  def predict(self, data):
    predictions = self.model.predict(data)
    return predictions, None
