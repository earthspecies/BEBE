from sklearn.mixture import GaussianMixture
from sklearn.metrics import normalized_mutual_info_score as nmi
from sklearn.metrics import accuracy_score
import yaml
import numpy as np

class gmm():
  def __init__(self, config):
    self.config = config
    self.model = GaussianMixture(n_components = self.config['num_components'], verbose = 2)
    
    metadata_fp = config['metadata_fp']
    with open(metadata_fp) as file:
      self.metadata = yaml.load(file, Loader=yaml.FullLoader)
      
    cols_included_bool = [x in self.config['input_vars'] for x in self.metadata['clip_column_names']] 
    self.cols_included = [i for i, x in enumerate(cols_included_bool) if x]
    
    
  def fit(self):
    ## get data. assume stored in memory for now
    train_fps = self.config['train_data_fp']
    
    train_data = [np.load(fp)[:, self.cols_included] for fp in train_fps]
    train_data = np.concatenate(train_data, axis = 0)

    self.model.fit(train_data)
  
  def predict(self, filepath):
    data = np.load(filepath)[:, self.cols_included]
    predictions = self.model.predict(data)
    return predictions
  