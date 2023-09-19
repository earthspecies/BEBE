from sklearn.decomposition import PCA
import numpy as np
import scipy.signal as signal
import pandas as pd

def load_wavelet_transformed_data(self, filepath, downsample):
  # perform wavelet transform during loading
  t, dt = np.linspace(0, 1, self.n_wavelets, retstep=True)
  fs = 1/dt
  freq = np.linspace(1, fs/2, self.n_wavelets)
  widths = self.morlet_w*fs / (2*freq*np.pi)

  data = pd.read_csv(filepath, delimiter = ',', header = None).values[:, self.cols_included]
  data = static_acc_filter(data, self.config)

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
  return transformed.astype(np.float32)

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
    """
    Extract static component and dynamic component from accelerometer data, return all channels with static/dynamic separated
    See obda.m available in the Animal Tags toolbox at animaltags.org
    Input
    -----
      series: [time, channels]. The number of channels is equal to config['input_vars']
      config: dict
    """
    
    sr = config['metadata']['sr']
    channels_to_process = [('Acc' in x) for x in config['input_vars']]
    static_acc_cutoff_freq = config['static_acc_cutoff_freq']
    
    if static_acc_cutoff_freq == 0:
      return series
    
    else: 
      n = 5 * np.round( sr / static_acc_cutoff_freq )
      new_series = []
      for i in range(series.shape[1]):
        s = series[:, i, None]
        if channels_to_process[i]:
          dynamic_component = fir_nodelay_highpass(s, n, static_acc_cutoff_freq/(sr/2))
          static_component = s - dynamic_component
          new_series.append(np.concatenate((static_component, dynamic_component), axis=-1))
        else:
          new_series.append(s)

      new_series = np.concatenate(new_series, axis = -1)

      return new_series

def fir_nodelay_highpass(s, n, fc):
    """
    High-pass delay-free filtering using a linear-phase (symmetric) FIR filter followed by group delay correction 
    See fir_nodelay.m available in the Animal Tags toolbox at animaltags.org
    Input
    -----
      s: [time, channels]. The number of channels is equal to config['input_vars']
      n: length of the symmetric FIR filter to use in units of input samples
      fc: filter cut off frequency relative to 1=Nyquist
    Returns
    -------
      y is the filtered signal with the same size as series (dynamic acceleration component)
    """
    n = int(np.floor(n/2)*2)
    h = signal.firwin(n + 1, fc, window="hamming", pass_zero=False) #scipy uses numtaps whereas matlab fir1 uses filter order
    noffs = int(np.floor(n/2))
    pad0 = s[noffs-1:0:-1,:]
    pad1 = s[-2:-noffs-2:-1,:]
    padded_series = np.concatenate( [pad0, s, pad1] , axis=0)
    y = signal.lfilter(h, 1.0, padded_series, axis=0)
    y = y[(n-1):(s.shape[0]+n-1), :]
    return y
  
def normalize_acc_magnitude(series, config):
  """
  On the fly normalize acc channels so mean field strength = 1.
  Assumes only one set of 3 Acc channels
  """
  channels_to_process = [('Acc' in x) for x in config['input_vars']]
  
  series_sub = series[:,channels_to_process]
  field_strength = np.mean(np.sqrt(series_sub[:,0] ** 2 + series_sub[:,1] ** 2 + series_sub[:,2] ** 2))
  
  for i in channels_to_process:
    series[:,i] = series[:,i] / (field_strength + 1e-6)
    
  return series
  