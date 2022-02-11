from sklearn.mixture import GaussianMixture
import yaml
import numpy as np
import pickle
import os

class gmm():
  def __init__(self, config):
    self.config = config
    self.model_config = config['gmm_config']
    self.model = GaussianMixture(n_components = self.config['num_clusters'], verbose = 2, max_iter = self.model_config['max_iter'], n_init = self.model_config['n_init'])
    self.metadata = config['metadata']
      
    cols_included_bool = [x in self.config['input_vars'] for x in self.metadata['clip_column_names']] 
    self.cols_included = [i for i, x in enumerate(cols_included_bool) if x]
  
  def load_model_inputs(self, filepath):
    return np.load(filepath)[:, self.cols_included]
    
  def fit(self):
    ## get data. assume stored in memory for now
    train_fps = self.config['train_data_fp']
    
    train_data = [self.load_model_inputs(fp) for fp in train_fps]
    train_data = np.concatenate(train_data, axis = 0)

    self.model.fit(train_data)
    
  def save(self):
    target_fp = os.path.join(self.config['final_model_dir'], "final_model.pickle")
    with open(target_fp, 'wb') as f:
      pickle.dump(self, f)
  
  def predict(self, data):
    predictions = self.model.predict(data)
    return predictions
  
  def predict_from_file(self, fp):
    inputs = self.load_model_inputs(fp)
    predictions = self.predict(inputs)
    return predictions
