# ## class to normalize and whiten data

from sklearn.decomposition import PCA
import numpy as np
  
class whitener_standalone():
  def __init__(self, n_components = 'mle'):
    super(whitener_standalone, self).__init__()
    
    self.whitener = None
    self.data_means = None
    self.data_std = None
    self.n_components = n_components
    
  def fit_transform(self, data):
    
    print("Whitening data")
    self.data_means = np.mean(data, axis = 0, keepdims = True)
    self.data_std = np.std(data, axis = 0, keepdims = True)
    
    data = data - self.data_means
    data = data / self.data_std    
    
    print("computing whitening transform")
    pca = PCA(n_components = self.n_components, whiten = True)
    transformed_data = pca.fit_transform(data)  
    self.whitener = pca
    print("whitened using %d components out of %d input dimensions" % (pca.n_components_ , np.shape(data)[1]))
    
    return transformed_data
  
  def transform(self, data):
    data = data - self.data_means
    data = data / self.data_std
    whitened_data = self.whitener.transform(data)
    
    return whitened_data
