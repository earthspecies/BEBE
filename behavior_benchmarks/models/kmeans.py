### KMeans class, primarily used for whitening data in a way that functions in the repo as do other methods of extracting latents

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import yaml
import numpy as np
import pickle
import os

class kmeans():
  def __init__(self, config):
    self.config = config
    self.model_config = config['kmeans_config']
    self.read_latents = config['read_latents']
    self.model = KMeans(n_clusters = self.config['num_clusters'], verbose = 0, max_iter = self.model_config['max_iter'], n_init = self.model_config['n_init'])
    
    self.metadata = config['metadata']
      
    cols_included_bool = [x in self.config['input_vars'] for x in self.metadata['clip_column_names']] 
    self.cols_included = [i for i, x in enumerate(cols_included_bool) if x]
    
    self.encoder = None
  
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
    
    print("computing whitening transform")
    pca = PCA(whiten = True)
    train_data = pca.fit_transform(train_data)  
    
    self.encoder = pca
    print("fitting kmeans")
    self.model.fit(train_data)
    
  def save(self):
    target_fp = os.path.join(self.config['final_model_dir'], "final_model.pickle")
    with open(target_fp, 'wb') as f:
      pickle.dump(self, f)
  
  def predict(self, data):
    whitened_data = self.encoder.transform(data)
    predictions = self.model.predict(whitened_data)
    return predictions, whitened_data
  
  def predict_from_file(self, fp):
    inputs = self.load_model_inputs(fp, read_latents = self.read_latents)
    predictions, latents = self.predict(inputs)
    return predictions, latents
