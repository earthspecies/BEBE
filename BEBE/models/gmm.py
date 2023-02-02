from sklearn.mixture import GaussianMixture
from BEBE.models.model_superclass import BehaviorModel
import yaml
import numpy as np
import pickle
import os

class gmm(BehaviorModel):
  def __init__(self, config):
    super(gmm, self).__init__(config)
    self.subselect_proportion = self.model_config['subselect_proportion']
    self.model = GaussianMixture(n_components = self.config['num_clusters'], verbose = 2, max_iter = self.model_config['max_iter'], n_init = self.model_config['n_init'])
    
  def fit(self):
    if self.read_latents:
      dev_fps = self.config['dev_data_latents_fp']
    else:
      dev_fps = self.config['dev_data_fp']
    
    dev_data = [self.load_model_inputs(fp, read_latents = self.read_latents) for fp in dev_fps]
    dev_data = np.concatenate(dev_data, axis = 0)
    
    if self.subselect_proportion < 1.:
        rng = np.random.default_rng()
        total_samples = np.shape(dev_data)[0]
        n_to_choose = int(self.subselect_proportion * total_samples)
        dev_data = rng.choice(dev_data, n_to_choose, replace=False)
        print("Subselecting %d out of %d total samples" % (n_to_choose, total_samples))

    self.model.fit(dev_data)
    
  def save(self):
    target_fp = os.path.join(self.config['final_model_dir'], "final_model.pickle")
    with open(target_fp, 'wb') as f:
      pickle.dump(self, f)
  
  def predict(self, data):
    predictions = self.model.predict(data)
    return predictions, None
