# ## class to normalize and whiten data

from sklearn.decomposition import PCA
import yaml
import numpy as np
import pickle
import os

class whiten():
  def __init__(self, config):
    self.config = config
    self.model_config = config['whiten_config']
    self.read_latents = config['read_latents']
    
    self.metadata = config['metadata']
      
    cols_included_bool = [x in self.config['input_vars'] for x in self.metadata['clip_column_names']] 
    self.cols_included = [i for i, x in enumerate(cols_included_bool) if x]
    
    self.whitener = None
    self.data_means = None
    self.data_std = None
  
  def load_model_inputs(self, filepath, read_latents = False):
    if read_latents:
      return np.load(filepath)
    else:
      return np.load(filepath)[:, self.cols_included]
    
  def fit(self):
    ## get data. assume stored in memory for now
    if self.read_latents:
      train_fps = self.config['train_data_latents_fp']
    else:
      train_fps = self.config['train_data_fp']
    
    train_data = [self.load_model_inputs(fp, read_latents = self.read_latents) for fp in train_fps]
    train_data = np.concatenate(train_data, axis = 0)
    
    print("regularizing data")
    self.data_means = np.mean(train_data, axis = 0, keepdims = True)
    self.data_std = np.std(train_data, axis = 0, keepdims = True)
    
    train_data = train_data - self.data_means
    train_data = train_data / self.data_std    
    
    print("computing whitening transform")
    pca = PCA(whiten = True)
    pca.fit(train_data)  
    self.whitener = pca
    
  def save(self):
    target_fp = os.path.join(self.config['final_model_dir'], "final_model.pickle")
    with open(target_fp, 'wb') as f:
      pickle.dump(self, f)
  
  def predict(self, data):
    data = data - self.data_means
    data = data / self.data_std
    whitened_data = self.whitener.transform(data)
    
    predictions_placeholder = np.zeros(np.shape(data)[0], dtype = int)
    return predictions_placeholder, whitened_data
  
  def predict_from_file(self, fp):
    inputs = self.load_model_inputs(fp, read_latents = self.read_latents)
    predictions, latents = self.predict(inputs)
    return predictions, latents
