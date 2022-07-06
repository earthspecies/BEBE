# ## class to normalize and whiten data

from sklearn.decomposition import PCA
import yaml
import numpy as np
import pickle
import os
from BEBE.models.model_superclass import BehaviorModel

# class whiten(BehaviorModel):
#   def __init__(self, config):
#     super(whiten, self).__init__(config)
    
#     self.whitener = None
#     self.data_means = None
#     self.data_std = None
    
#   def fit(self):
#     ## get data. assume stored in memory for now
#     if self.read_latents:
#       dev_fps = self.config['dev_data_latents_fp']
#     else:
#       dev_fps = self.config['dev_data_fp']
    
#     dev_data = [self.load_model_data(fp, read_latents = self.read_latents) for fp in dev_fps]
#     dev_data = np.concatenate(dev_data, axis = 0)
    
#     print("regularizing data")
#     self.data_means = np.mean(dev_data, axis = 0, keepdims = True)
#     self.data_std = np.std(dev_data, axis = 0, keepdims = True)
    
#     dev_data = dev_data - self.data_means
#     dev_data = dev_data / self.data_std    
    
#     print("computing whitening transform")
#     pca = PCA(n_components = 'mle', whiten = True)
#     pca.fit(dev_data)  
#     self.whitener = pca
#     print("whitened using %d components out of %d input dimensions" % (pca.n_components_ , np.shape(dev_data)[1]))
    
#   def save(self):
#     target_fp = os.path.join(self.config['final_model_dir'], "final_model.pickle")
#     with open(target_fp, 'wb') as f:
#       pickle.dump(self, f)
  
#   def predict(self, data):
#     data = data - self.data_means
#     data = data / self.data_std
#     whitened_data = self.whitener.transform(data)
    
#     predictions_placeholder = np.zeros(np.shape(data)[0], dtype = int)
#     return predictions_placeholder, whitened_data
  
class whitener_standalone():
  def __init__(self):
    super(whitener_standalone, self).__init__()
    
    self.whitener = None
    self.data_means = None
    self.data_std = None
    
  def fit_transform(self, data):
    
    print("Whitening data")
    self.data_means = np.mean(data, axis = 0, keepdims = True)
    self.data_std = np.std(data, axis = 0, keepdims = True)
    
    data = data - self.data_means
    data = data / self.data_std    
    
    print("computing whitening transform")
    pca = PCA(n_components = 'mle', whiten = True)
    transformed_data = pca.fit_transform(data)  
    self.whitener = pca
    print("whitened using %d components out of %d input dimensions" % (pca.n_components_ , np.shape(data)[1]))
    
    return transformed_data
  
  def transform(self, data):
    data = data - self.data_means
    data = data / self.data_std
    whitened_data = self.whitener.transform(data)
    
    return whitened_data
