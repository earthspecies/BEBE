### KMeans class

from sklearn.cluster import KMeans
from BEBE.models.model_superclass import BehaviorModel
from BEBE.models.whiten import whitener_standalone
import yaml
import numpy as np
import pickle
import os
import pandas as pd
import scipy.signal as signal


class kmeans(BehaviorModel):
  def __init__(self, config):
    super(kmeans, self).__init__(config)
    self.model = KMeans(n_clusters = self.config['num_clusters'], verbose = 0, max_iter = self.model_config['max_iter'], n_init = self.model_config['n_init'], random_state = self.config['seed'])
    
    self.whiten = self.model_config['whiten']
    self.n_components = self.model_config['n_components']
    self.whitener = whitener_standalone(n_components = self.n_components)
    
    self.wavelet_transform = self.model_config['wavelet_transform']
    
    # wavelet transform: specific settings
    self.morlet_w = self.model_config['morlet_w']
    self.n_wavelets = self.model_config['n_wavelets']
    self.downsample = self.model_config['downsample']
    
  def load_model_inputs(self, filepath, downsample = 1):
    
    if self.wavelet_transform:
      # perform wavelet transform during loading
      t, dt = np.linspace(0, 1, self.n_wavelets, retstep=True)
      fs = 1/dt
      freq = np.linspace(1, fs/2, self.n_wavelets)
      widths = self.morlet_w*fs / (2*freq*np.pi)

      data = pd.read_csv(filepath, delimiter = ',', header = None).values[:, self.cols_included]

      axes = np.arange(0, np.shape(data)[1])
      transformed = []
      for axis in axes:
          sig = data[:, axis]
          sig = (sig - np.mean(sig)) / (np.std(sig) + 1e-6) # normalize each channel independently
          if downsample > 1:
              transformed.append(np.abs(signal.cwt(sig, signal.morlet2, widths, w=self.morlet_w))[:, ::downsample])
          else:
              transformed.append(np.abs(signal.cwt(sig, signal.morlet2, widths, w=self.morlet_w)))

      transformed = np.concatenate(transformed, axis = 0)
      transformed = np.transpose(transformed)
      return transformed
    
    else:
      return pd.read_csv(filepath, delimiter = ',', header = None).values[:, self.cols_included]

    
  def fit(self):
    ## get data. assume stored in memory for now
    train_fps = self.config['train_data_fp']
    
    train_data = [self.load_model_inputs(fp, downsample = self.downsample) for fp in train_fps]
    train_data = np.concatenate(train_data, axis = 0)
    
    if self.whiten:
      train_data = self.whitener.fit_transform(train_data)
    
    print("fitting kmeans")
    self.model.fit(train_data)
    
  def save(self):
    target_fp = os.path.join(self.config['final_model_dir'], "final_model.pickle")
    with open(target_fp, 'wb') as f:
      pickle.dump(self, f)
  
  def predict(self, data):
    if self.whiten:
      data = self.whitener.transform(data)                         
    predictions = self.model.predict(data)
    return predictions, None
