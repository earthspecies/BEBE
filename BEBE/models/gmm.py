from sklearn.mixture import GaussianMixture
from BEBE.models.model_superclass import BehaviorModel
import yaml
import numpy as np
import pickle
import os

class gmm(BehaviorModel):
  def __init__(self, config):
    super(gmm, self).__init__(config)
    self.downsample = self.model_config['downsample']
    self.model = GaussianMixture(n_components = self.config['num_clusters'], verbose = 2, max_iter = self.model_config['max_iter'], n_init = self.model_config['n_init'], random_state = self.config['seed'])
    
  def fit(self):
    train_fps = self.config['train_data_fp']
    
    train_data = [self.load_model_inputs(fp) for fp in train_fps]
    train_data = np.concatenate(train_data, axis = 0)
    
    if self.downsample > 1:
        train_data = train_data[::self.downsample, :]

    self.model.fit(train_data)
    
  def save(self):
    target_fp = os.path.join(self.config['final_model_dir'], "final_model.pickle")
    with open(target_fp, 'wb') as f:
      pickle.dump(self, f)
  
  def predict(self, data):
    predictions = self.model.predict(data)
    return predictions, None
