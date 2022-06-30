# Motionmapper-esque. Uses UMAP for dimensionality reduction rather than tsne
import sys
import os
import glob
import yaml
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import scipy.signal as signal
from scipy import ndimage as ndi
import scipy
import sklearn
import umap
from behavior_benchmarks.models.model_superclass import BehaviorModel
import pickle
from skimage.transform import resize

from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from scipy.sparse import csc_matrix

class umapper(BehaviorModel):
  def __init__(self, config):
    super(umapper, self).__init__(config)
    sr = config['metadata']['sr']
    self.morlet_w = self.model_config['morlet_w']
    self.n_wavelets = self.model_config['n_wavelets']
    self.image_border = self.model_config['image_border']
    self.image_size = self.model_config['image_size']
    self.n_watershed_trials = self.model_config['n_watershed_trials']
    self.downsample = self.model_config['downsample']
    n_neighbors = self.model_config['n_neighbors']
    min_dist = self.model_config['min_dist']
    self.num_clusters = self.config['num_clusters']
    
    
    low_freq_cols_bool = [x in self.model_config['low_freq_cols'] for x in self.metadata['clip_column_names']] # which columns to not apply wavelet transform to
    self.low_freq_cols = [i for i, x in enumerate(low_freq_cols_bool) if x]
    
    high_freq_cols_bool = [(x in self.config['input_vars'] and x not in self.model_config['low_freq_cols']) for x in self.metadata['clip_column_names']] # which columns to apply wavelet transform to
    self.high_freq_cols = [i for i, x in enumerate(high_freq_cols_bool) if x]
    
    # initialize umap
    self.reducer = umap.UMAP(
        n_neighbors = n_neighbors,
        min_dist = min_dist,
        metric = 'symmetric_kl',
        verbose = True
    )
    
  def load_model_inputs(self, filepath, read_latents = False):
    # perform wavelet transform during loading
    # Perform morlet wavelet transform
    t, dt = np.linspace(0, 1, self.n_wavelets, retstep=True)
    fs = 1/dt
    freq = np.linspace(1, fs/2, self.n_wavelets)
    widths = self.morlet_w*fs / (2*freq*np.pi)
    
    if read_latents:
      raise NotImplementedError
    else:
      #data = np.genfromtxt(filepath, delimiter = ',')[:, self.cols_included]
      low_freq_data = np.genfromtxt(filepath, delimiter = ',')[:, self.low_freq_cols]
      high_freq_data = np.genfromtxt(filepath, delimiter = ',')[:, self.high_freq_cols]
    
    axes = np.arange(0, np.shape(high_freq_data)[1])
    transformed = []
    for axis in axes:
        sig = high_freq_data[:, axis]
        sig = (sig - np.mean(sig)) / (np.std(sig) + 1e-6) # normalize
        transformed.append(np.abs(signal.cwt(sig, signal.morlet2, widths, w=self.morlet_w)))
        
    axes = np.arange(0, np.shape(low_freq_data)[1])
    for axis in axes:
        sig = low_freq_data[:, axis]
        #sig = (sig - np.mean(sig)) / (np.std(sig) + 1e-6) # normalize
        transformed.append(np.expand_dims(sig, 0)) # do not transform low frequency data

    transformed = np.concatenate(transformed, axis = 0)
    transformed = np.transpose(transformed)
      
    return transformed
    
  def fit(self):
    ## get data. assume stored in memory for now
    if self.read_latents:
      dev_fps = self.config['dev_data_latents_fp']
    else:
      dev_fps = self.config['dev_data_fp']
    
    # load as wavelets
    dev_data = [self.load_model_inputs(fp, read_latents = self.read_latents)[::self.downsample, :] for fp in dev_fps]
    dev_data = np.concatenate(dev_data, axis = 0)
    
    # normalize and record normalizing constant
    normalize_denom = np.sum(dev_data, axis = 1, keepdims = True)
    dev_data = dev_data / (normalize_denom + 1e-6)
    
    # fit umap
    y = self.reducer.fit_transform(dev_data)
    
    # learn how to rescale into useful image
    self.translation = np.amin(y, axis = 0)
    yq = y - self.translation # translate
    self.scale_factor = (self.image_size - 2*self.image_border) / np.amax(yq)
    yq = (yq * self.scale_factor).astype(int) + self.image_border
    yq = np.maximum(yq, 0)
    yq = np.minimum(yq, self.image_size-1)
    
    # Turn into dense array format
    data = np.ones_like(yq[:, 0]) #every entry in the list yq contributes 1
    row = yq[:, 0]
    col = yq[:, 1]
    y_as_array = scipy.sparse.csc_matrix((data, (row, col)), shape=(self.image_size, self.image_size)).toarray()
    y_as_array = y_as_array / np.amax(y_as_array)
    
    # Do a binary search to take gaussian filter that gives us at most num_clusters clusters
    # We assume that higher sigma in gaussian filter results in fewer watershed basins

    sigma_max = None #lowest upper bound on sigma that WILL work (<= desired number of clusters)
    sigma_min = 1. #greatest lower bound on sigma that WON'T work (leads to too many clusters)...at first we aren't sure if this is a good lower bound
    good_sigma_min = False
    best_label_image = None

    # first search for an upper and lower bound on sigma
    print("Trying different gaussian kernels")
    while True:
        sigma = 2 * sigma_min
        print("trying with sigma %3.3f" % sigma)
        image = ndi.gaussian_filter(y_as_array, sigma)

        coords = peak_local_max(image, footprint=np.ones((3, 3)))
        mask = np.zeros(image.shape, dtype=bool)
        mask[tuple(coords.T)] = True
        markers, _ = ndi.label(mask)
        labels = watershed(-image, markers) - 1
        n_discovered_labels = np.amax(labels) + 1

        if n_discovered_labels > self.num_clusters:
            good_sigma_min = True
            sigma_min = sigma
        if n_discovered_labels <= self.num_clusters:
            if good_sigma_min:
              sigma_max = sigma
              best_label_image = labels
              break
            else:
              sigma_min = sigma_min/2. # if starting sigma is too high, we reduce and try again

    # Search between sigma_min and sigma_max
    print("doing final binary search for gaussian kernel")
    for i in tqdm(range(self.n_watershed_trials)):
        sigma = (sigma_max + sigma_min)/2
        image = ndi.gaussian_filter(y_as_array, sigma)

        coords = peak_local_max(image, footprint=np.ones((3, 3)))
        mask = np.zeros(image.shape, dtype=bool)
        mask[tuple(coords.T)] = True
        markers, _ = ndi.label(mask)
        labels = watershed(-image, markers) - 1
        n_discovered_labels = np.amax(labels) + 1

        if n_discovered_labels > self.num_clusters:
            sigma_min = sigma
        if n_discovered_labels <= self.num_clusters:
            sigma_max = sigma
            best_label_image = labels

    # Finally, use best found sigma
    self.label_image = best_label_image

    fig, axes = plt.subplots(ncols=3, figsize=(12, 4), sharex=True, sharey=True)
    ax = axes.ravel()

    ax[0].scatter(yq[:,1], yq[:, 0].T, s = 1)
    ax[0].set_title('UMAP')
    ax[1].imshow(image)
    ax[1].set_title('UMAP + Gaussian kernel')
    ax[2].imshow(labels)
    ax[2].scatter(yq[:,1], yq[:, 0].T, s = 1, c = 'white')
    ax[2].set_title('Watershed Transform')

    for a in ax:
        a.set_axis_off()

    fp = os.path.join(self.config['output_dir'], 'UMAP_vis.png')
    plt.savefig(fp)
    
  def save(self):
    target_fp = os.path.join(self.config['final_model_dir'], "final_model.pickle")
    with open(target_fp, 'wb') as f:
      pickle.dump(self, f)
  
  def predict(self, data):
    # normalize
    normalize_denom = np.sum(data, axis = 1, keepdims = True)
    data = data / (normalize_denom + 1e-6)
    
    data_downsampled = data[::self.downsample, :]
    
    # fit umap
    y = self.reducer.transform(data_downsampled)
    
    # rescale into useful image
    yq = y - self.translation # translate
    yq = (yq * self.scale_factor).astype(int) + self.image_border # rescale
    yq = np.maximum(yq, 0) #crop
    yq = np.minimum(yq, self.image_size-1)
    
    predictions = self.label_image[yq[:,0], yq[:, 1]]
    
    predictions = resize(predictions, (np.shape(data)[0],), order=0, mode='constant')
    
    return predictions, None
