### KMeans class

from sklearn.cluster import KMeans
from BEBE.models.model_superclass import BehaviorModel
from BEBE.models.whiten import whitener_standalone
import yaml
import numpy as np
import pickle
import os

class kmeans(BehaviorModel):
  def __init__(self, config):
    super(kmeans, self).__init__(config)

    self.model = KMeans(n_clusters = self.config['num_clusters'], verbose = 0, max_iter = self.model_config['max_iter'], n_init = self.model_config['n_init'])
    
    self.whiten = self.model_config['whiten']
    self.n_components = self.model_config['n_components']
    self.whitener = whitener_standalone(n_components = self.n_components)
    
  def fit(self):
    ## get data. assume stored in memory for now
    if self.read_latents:
      dev_fps = self.config['dev_data_latents_fp']
    else:
      dev_fps = self.config['dev_data_fp']
    
    dev_data = [self.load_model_inputs(fp, read_latents = self.read_latents) for fp in dev_fps]
    dev_data = np.concatenate(dev_data, axis = 0)
    
    if self.whiten:
      dev_data = self.whitener.fit_transform(dev_data)
    
    print("fitting kmeans")
    self.model.fit(dev_data)
    
  def save(self):
    target_fp = os.path.join(self.config['final_model_dir'], "final_model.pickle")
    with open(target_fp, 'wb') as f:
      pickle.dump(self, f)
  
  def predict(self, data):
    if self.whiten:
      data = self.whitener.transform(data)
                         
    predictions = self.model.predict(data)
    return predictions, None
