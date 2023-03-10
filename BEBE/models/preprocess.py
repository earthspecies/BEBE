from sklearn.decomposition import PCA
import numpy as np
import scipy.signal as signal

# class to normalize and whiten data
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
  
def static_acc_filter(series, config):
    # extract static component
    # series: [time, channels]. The number of channels is equal to config['input_vars']
    # low_cutoff_freq: float
    
    sr = config['metadata']['sr']
    channels_to_process = [('Acc' in x) for x in config['input_vars']]
    filter_order = 10
    static_acc_cutoff_freq = config['static_acc_cutoff_freq']
    
    
    if static_acc_cutoff_freq == 0:
      return series
    
    else: 
      sos = signal.butter(filter_order, static_acc_cutoff_freq, 'low', fs=sr, output='sos')
      new_series = []
      for i in range(np.shape(series)[1]):
        if channels_to_process[i]:
          s = series[:, i]
          low_passed_s = signal.sosfilt(sos, s)
          remaining_s = s - low_passed_s
          both_s = np.stack([low_passed_s, remaining_s], axis = -1)
          new_series.append(both_s)
        else:
          new_series.append(series[:, i:i+1])
      new_series = np.concatenate(new_series, axis = -1)
      return new_series
        