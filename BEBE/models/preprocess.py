from sklearn.decomposition import PCA
import numpy as np
import scipy.signal as signal
import pandas as pd

def compute_wavelets(x, actual_sr, n_wavelets, C_min, C_max, morlet_w):

    ## Computation following Sakamoto

    # Morlet wavlet by scipy: exp(1j*w*x/s) * exp(-0.5*(x/s)**2) * pi**(-0.25) * sqrt(1/s)
    # note this differs from the one in Sakamoto by a factor of sqrt(1/s)
    # This changes the amplitude of the wavelet (in each frequency band), but not the frequency

    # Sakamoto computes convolution with morlet_w = 10 and s = k*lambda
    # where
    # k = (morlet_w + np.sqrt(2 + morlet_w**2)) / (4 * np.pi)
    # Sakamoto additionally multiplies by some normalizing factors Ce/(k*lambda), which depend only on lambda (i.e. s)
    # Therefore, up to per-frequency-band rescaling, the computation to match Sakamoto is:
    
    if C_min is None:
        C_min = 2 / actual_sr # set max freq to nyquist

    Lambda = C_min * 2**(np.arange(n_wavelets) * np.log2(C_max/C_min) / (n_wavelets-1))

    k = (morlet_w + np.sqrt(2 + morlet_w**2)) / (4 * np.pi)
    widths = k * Lambda #units are sec
    widths = widths * actual_sr

    # ## Verify 
    # # Reversing computation above
    # scale_to_freq = lambda s: (actual_sr * k) / s
    # y_axis_freqs = scale_to_freq(widths)
    # # print("Intended freqs: ", freq)
    # # print(f"Actual freqs under morlet2, with sr={actual_sr}Hz: ", y_axis_freqs)

    y = np.abs(signal.cwt(x, signal.morlet2, widths, w=morlet_w))
    return y

def load_wavelet_transformed_data(self, filepath, downsample):
  # perform wavelet transform during loading

  data = pd.read_csv(filepath, delimiter = ',', header = None).values[:, self.cols_included]

  axes = np.arange(0, np.shape(data)[1])
  transformed = []
  for axis in axes:
      sig = data[:, axis]
      if self.model_config['per_channel_normalize']:
          sig = (sig - np.mean(sig)) / (np.std(sig) + 1e-6) # normalize each channel independently
      if downsample > 1:
          transformed.append(compute_wavelets(sig, self.metadata['sr'], self.model_config['n_wavelets'], self.model_config['C_min'], self.model_config['C_max'], self.model_config['morlet_w'])[:, ::downsample])
      else:
          transformed.append(compute_wavelets(sig, self.metadata['sr'], self.model_config['n_wavelets'], self.model_config['C_min'], self.model_config['C_max'], self.model_config['morlet_w']))

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
  Assumes triaxial acc sensor channels are not shuffled, if there are > 1 triaxial acc sensor
  """
  
  # check we have correct number of acc channels
  channels_to_process = [('Acc' in x) for x in config['input_vars']]
  assert sum(channels_to_process) % 3 == 0
  
  # go through groups successively, and normalize
  acc_group_idxs = []
  for i, x in enumerate(config['input_vars']):
    if 'Acc' in x:
      acc_group_idxs.append(i)
    if len(acc_group_idxs) == 3:
      series_sub = series[:,acc_group_idxs]
      field_strength = np.mean(np.sqrt(series_sub[:,0] ** 2 + series_sub[:,1] ** 2 + series_sub[:,2] ** 2))
      series[:,acc_group_idxs] = series[:,acc_group_idxs] / (field_strength + 1e-6)
      acc_group_idxs = []
    
  return series
  