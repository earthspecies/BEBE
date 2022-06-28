# ## Random assignment as a control
from behavior_benchmarks.models.model_superclass import BehaviorModel
import numpy as np
import pickle
import os

class random(BehaviorModel):
  def __init__(self, config):
    super(random, self).__init__(config)
    self.rng = np.random.default_rng()
    self.num_clusters = self.config['num_clusters']
    
  def fit(self):
    pass
    
  def save(self):
    target_fp = os.path.join(self.config['final_model_dir'], "final_model.pickle")
    with open(target_fp, 'wb') as f:
      pickle.dump(self, f)
  
  def predict(self, data):
    n_samples = np.shape(data)[0]
    predictions = self.rng.integers(low = 0, high = self.num_clusters, size = n_samples)
    return predictions, None
  