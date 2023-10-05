import tqdm
import os
import pickle
import random
import pandas as pd

import torch
import numpy as np
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader, Subset

from BEBE.models.model_superclass import BehaviorModel
from BEBE.models.preprocess import static_acc_filter, normalize_acc_magnitude, load_wavelet_transformed_data

def circular_variance(angles):
    """ Compute circular variance. angles.shape = (n_angles,) """
    # first Compute mean resultant vector length for circular data
    r = torch.sum(torch.exp(1j*angles));
    r = torch.abs(r) / len(angles)
    return 1 - r

def nathan_basic_features(multi_channel_data, epsilon = 1e-12):
    """ Moments, autocorrelation, and trend from feature set in Nathan et al., 2012 """
    features = []
    #  mean
    mu = torch.mean(multi_channel_data, dim=0, keepdim=True)
    features.append(mu)
    #  standard deviation
    std = torch.std(multi_channel_data, dim=0, keepdim=True)
    features.append(std)
    #  skewness
    zscores = (multi_channel_data - mu) / ( std + epsilon)
    features.append(torch.mean(torch.pow(zscores, 3.0), dim=0, keepdim=True))
    #  kurtosis
    features.append(torch.mean(torch.pow(zscores, 4.0), dim=0, keepdim=True) - 3.0 )
    #  maximum value
    mx = torch.max(multi_channel_data, dim=0, keepdim=True).values
    features.append(mx)
    #  minimum value
    mn = torch.min(multi_channel_data, dim=0, keepdim=True).values
    features.append(mn)
    #  autocorrelation for a displacement of one measurement
    numerator = (multi_channel_data[1:, :] - mu) * (multi_channel_data[:-1, :] - mu)
    numerator = numerator.mean(dim=0, keepdim=True)
    ac = numerator / (std**2 + epsilon)
    features.append(ac)
    #  trend: linear regression through the data
    x = torch.arange(0, multi_channel_data.shape[0], dtype=torch.float32)
    trend = torch.sum((x[:, None] - x.mean()) * (multi_channel_data - mu), dim = 0, keepdim=True)
    features.append(trend)
    return torch.cat([f.squeeze(dim=0) for f in features])

def triaxial_correlation_features(triaxial_data):
    """ Correlations from feature set in Nathan et al., 2012 """
    features = []
    mu = triaxial_data.mean(0)
    # Three pairwise correlations: xy, xz, yz 
    for (a_idx, b_idx) in [(0, 1), (0, 2), (1, 2)]:
       a = triaxial_data[:, a_idx] - mu[a_idx]
       b = triaxial_data[:, b_idx] - mu[b_idx]
       numerator = torch.sum(a * b, dim = 0)
       denominator = torch.sqrt(torch.sum(b**2, dim = 0) * torch.sum(a**2, dim=0))
       features.append(numerator/(denominator + 1e-12))
    corr = torch.Tensor(features)
    return corr

def nathan_raw_features(triaxial_acc_data, epsilon = 1e-12):
    """
        Implement features based on raw acclerometer data from (Nathan et al., 2012)
        See https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3284320/
        
        Inputs
        ------
        triaxial_acc_data: torch Tensor
            raw xyz acclerometer values
            shape: (n_samples, n_channels=3)
    """
    # Include rms for basic features
    q = torch.sqrt( torch.sum(triaxial_acc_data ** 2, dim=-1, keepdims=True) )
    four_channel_data = torch.cat([triaxial_acc_data,q], dim=1)
    # Leave out OBDA, because requires dynamic accleration 
    features_basic = nathan_basic_features(four_channel_data)
    features_correlation = triaxial_correlation_features(triaxial_acc_data)
    # Circular variance of inclination for q axis = (θ=arccos[z/q])
    theta = torch.acos(triaxial_acc_data[:, 2] / (four_channel_data[:, 3] + epsilon))
    cvt = circular_variance(theta)
    # Circular variance of azimuth for q axis = (φ=arctan[y/x])
    psi = torch.atan(triaxial_acc_data[:, 1] / (triaxial_acc_data[:, 2] + epsilon))
    cvp = circular_variance(psi)
    cvs = torch.Tensor([cvt, cvp])
    return torch.cat([features_basic, features_correlation, cvs])

def mean_obda(dynamic_accleration):
    """ Compute mean overall dynamic body accleration over the window
        https://github.com/animaltags/tagtools_matlab/blob/main/processing/odba.m
    """
    obda = torch.mean(torch.sum(torch.abs(dynamic_accleration), dim=-1))[None]
    return obda

def split_into_triaxial(data_channels, n=3):
    """ If there are >n channels, group into sets of n channels. Assumes triaxial channels are listed one after another. """
    return [data_channels[:, i:i+n] for i in range(0, data_channels.shape[1], n)]

class Features(Dataset):
    
    def __init__(self, data, labels, individuals, train, temporal_window_samples, config):
        # Take features over a window 
        self.temporal_window_samples = temporal_window_samples 
        
        self.data = data # list of np arrays, each of shape [*, n_features] where * is the number of samples and varies between arrays
        if train:
            self.labels = labels # list of np arrays, each of shape [*,] where * is the number of samples and varies between arrays
        self.individuals = individuals # list of np arrays, each of shape [*,] where * is the number of samples and varies between arrays
        
        self.label_names = config['metadata']['label_names']
        self.num_classes = len(self.label_names)
        self.unknown_idx = self.label_names.index('unknown')
        self.data_points = sum([np.shape(x)[0] for x in self.data])
        print('Initialize dataloader. Datapoints %d' %self.data_points)
        self.input_vars = config['input_vars']
            
        self.data_start_indices = []
        counter = 0
        for x in self.data:
          assert np.shape(x)[0] > temporal_window_samples, "temporal_window_samples must be shorter than smallest example"
          self.data_start_indices.append(counter)
          counter = counter + np.shape(x)[0]
          
        assert counter == self.data_points
        self.data_start_indices = np.array(self.data_start_indices)
        self.data_stds = np.std(np.concatenate(self.data, axis = 0), axis = 0, keepdims = True) / 8
        self.num_channels = np.shape(self.data_stds)[1]
        self.rng = np.random.default_rng(config['seed'])
        self.train = train

        # Feature set setup
        self.feature_set = config['model_config']['feature_set']
        self.dynamic_accleration_only = config['metadata'].get('dynamic_acc_only', False)
        if self.feature_set == 'nathan2012':
            print("Using feature set from nathan et al. 2012")
            assert (config['static_acc_cutoff_freq'] > 0) or self.dynamic_accleration_only, "The Nathan et al 2012 feature set requires OBDA."
        elif self.feature_set == 'wavelet':
           print("Using wavelets as features")
        else:
            raise Exception(f"Feature set {self.feature_set} is not implemented.")
        
    def __len__(self):        
        return self.data_points
      
    def get_class_proportions(self):
        all_labels = np.concatenate(self.labels)
        counts = []
        for i in range(self.num_classes):
          counts.append(len(all_labels[all_labels == i]))
        total_labels = sum(counts[:self.unknown_idx] + counts[self.unknown_idx + 1:])
        weights = np.array([x/total_labels for x in counts], dtype = 'float')
        return torch.from_numpy(weights).type(torch.FloatTensor)
      
    def get_annotated_windows(self):
        # Go through data, make a list of the indices of windows which actually have annotations.
        # Useful for speeding up supervised train time for sparsely labeled datasets.
        indices_of_annotated_windows = []
        for clip_number, start_idx in enumerate(self.data_start_indices):
            indexes_within_x_with_known_annotation = np.where(self.labels[clip_number] != self.unknown_idx)[0]
            indices_of_annotated_windows.append(indexes_within_x_with_known_annotation + start_idx) 
        return np.concatenate(indices_of_annotated_windows)

    def balance_classes_by_individual(self):
        """ Returns a list of indexes which have annotations and which make the number of datapoints in each class equivalent to the smallest """
        all_individuals = np.concatenate(self.individuals)
        individual_idxs = np.unique(all_individuals)
        all_labels = np.concatenate(self.labels)
        indicies_of_annotated_and_balanced_windows = []
        for individual_idx in individual_idxs:
            b_indiv = (all_individuals == individual_idx)
            xs_indiv_class = []
            for c_idx in range(self.num_classes):
                if self.label_names[c_idx] == 'unknown': continue
                xs_indiv_class.append( np.where( b_indiv * (all_labels == c_idx) )[0] )
            smallest_class_per_indiv = min([len(xic) for xic in xs_indiv_class])
            balanced_xs_indiv_class = []
            for xic in xs_indiv_class:
                subsampled_idxs = self.rng.choice(xic, size=smallest_class_per_indiv, replace=False)
                balanced_xs_indiv_class.append(np.sort(subsampled_idxs))
            indicies_of_annotated_and_balanced_windows.append(np.concatenate(balanced_xs_indiv_class))
        return np.concatenate(indicies_of_annotated_and_balanced_windows)

    def compute_features(self, data_item):
        features = []
        if self.feature_set == "nathan2012":
            # Obtain acclerometer channels
            # See preprocess.py to see how static and dynamic accleration are treated
            if not self.dynamic_accleration_only:
                acc_channels_static_idxs = 2*np.where([('Acc' in _) for _ in self.input_vars])[0]
                acc_channels_dynamic_idxs = acc_channels_static_idxs + 1
                acc_channels_static = data_item[:, acc_channels_static_idxs]
                acc_channels_dynamic = data_item[:, acc_channels_dynamic_idxs]
                acc_channels_raw = acc_channels_static + acc_channels_dynamic
                all_acc_channel_idxs = np.concatenate([acc_channels_static_idxs,acc_channels_dynamic_idxs])
            else:
                # Only dynamic accleration is available, so use that in place of raw accleration for Nathan features
                acc_channels_dynamic_idxs = np.where([('Acc' in _) for _ in self.input_vars])[0]
                acc_channels_dynamic = data_item[:, acc_channels_dynamic_idxs]
                acc_channels_raw = acc_channels_dynamic
                all_acc_channel_idxs = acc_channels_dynamic_idxs
            acc_triaxial_dynamic = split_into_triaxial(acc_channels_dynamic)
            acc_triaxial_raw = split_into_triaxial(acc_channels_raw)
            # Obtain all other channels
            other_channel_idxs = [i for i in range(data_item.shape[1]) if i not in list(all_acc_channel_idxs)]
            # Obtain gyroscope channels, if any
            has_gyroscope = any([('Gyr' in _) for _ in self.input_vars])
            if has_gyroscope:
                previous_index = 0; gyr_channels_idxs = []
                for input_var in self.input_vars:
                    if 'Acc' in input_var:
                        # Account for static and dynamic channel
                        previous_index += 2
                    elif 'Gyr' in input_var:
                        gyr_channels_idxs.append(previous_index)
                        previous_index += 1
                    else:
                        previous_index += 1
                gyr_triaxial = split_into_triaxial(data_item[:,gyr_channels_idxs])

            ## Obtain acclerometer features
            for tia_channels in acc_triaxial_raw:
                features.append(nathan_raw_features(tia_channels))
                
            for dyn_channels in acc_triaxial_dynamic:
                features.append(mean_obda(dyn_channels))

            ## Acclerometer features are the only ones originally included in Nathan
            # Use Nathan basic features for all other channels
            if len(other_channel_idxs) > 0:
                features.append(nathan_basic_features(data_item[:, other_channel_idxs]))
            # Also use triaxial correlations for gyroscope channels
            if has_gyroscope:
                for gyr_channels in gyr_triaxial:
                    features.append(triaxial_correlation_features(gyr_channels))
        else:
            raise Exception(f"Feature set {self.feature_set} not implemented")
        
        return torch.cat(features)

    def __getitem__(self, index):
        """ Returns features of shape (n_features,) """
        clip_number = np.where(index >= self.data_start_indices)[0][-1] #which clip do I draw from?
        
        data_item = self.data[clip_number]
        middle = index - self.data_start_indices[clip_number]
        if self.train:
            labels_item = self.labels[clip_number][middle] 
        if self.feature_set == 'nathan2012':
          # Simply use a truncated window at the edges. TODO: should we pad instead?
          start = max(0, middle - self.temporal_window_samples//2)
          end = min(data_item.shape[0] - 1, middle + self.temporal_window_samples//2)
          windowed_data = data_item[start:end, :]
          features_item = self.compute_features(torch.Tensor(windowed_data))
        elif self.feature_set == 'wavelet':
          features_item = torch.Tensor(data_item[middle, :])
        else:
          raise Exception(f"Feature set {self.feature_set} is not implemented")
        
        if self.train:
            return features_item, labels_item
        else:
            return features_item
    
class ClassicBehaviorModel(BehaviorModel):
  def __init__(self, config):
    super().__init__(config)
    
    torch.manual_seed(self.config['seed'])
    random.seed(self.config['seed'])
    np.random.seed(self.config['seed'])
    
    # Get cpu or gpu device for training.
    self.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {self.device} device")
    self.num_workers = self.model_config.get('num_workers', 0)
    
    ## General Training Parameters
    self.downsizing_factor = self.model_config['downsizing_factor']
    # min val of 6 for temporal_window_samples prevents overly short windows at the edges (e.g., would lead to nan in std feature). TODO: should we truncate or pad at the edges?
    self.temporal_window_samples = max(int(np.ceil(self.model_config['context_window_sec'] * self.metadata['sr'])), 6)
    self.normalize = self.model_config['normalize']
    self.batch_size = self.model_config['batch_size']

    ## If wavelet features are used
    self.wavelet_transform = self.model_config['wavelet_transform']
    if self.wavelet_transform:
        # wavelet transform: specific settings
        assert self.model_config['feature_set'] == 'wavelet'
        self.morlet_w = self.model_config['morlet_w']
        self.n_wavelets = self.model_config['n_wavelets']
        self.downsample = self.model_config['downsample']
    
    # Dataset Parameters
    self.unknown_label = config['metadata']['label_names'].index('unknown')
    self.label_idx = [i for i, x in enumerate(self.metadata['clip_column_names']) if x == 'label'][0]
    self.individual_idx = [i for i, x in enumerate(self.metadata['clip_column_names']) if x == 'individual_id'][0]
    self.n_classes = len(self.metadata['label_names']) 
    self.n_features = self.get_n_features()
    self.balance_classes = config['balance_classes']

    # Specify in subclass
    self.model = None

  def get_n_features(self):
    # gets number of features after processing input data channels
    train_fps = self.config['train_data_fp']
    x = self.load_model_inputs(train_fps[0])
    dataset = Features([x], None, None, False, self.temporal_window_samples, self.config)
    return dataset[0].shape[0]
    
  def load_model_inputs(self, filepath):
    x = pd.read_csv(filepath, delimiter = ',', header = None).values[:, self.cols_included] #[n_samples, n_features]
    if self.wavelet_transform:
        x = load_wavelet_transformed_data(self, filepath, downsample=0)
    else:
        x = static_acc_filter(x, self.config)
        if self.normalize:
            x = normalize_acc_magnitude(x, self.config)
    return x
    
  def load_labels(self, filepath):
    labels = pd.read_csv(filepath, delimiter = ',', header = None).values[:, self.label_idx].astype(int)
    return labels 

  def load_individuals(self, filepath):
    individuals = pd.read_csv(filepath, delimiter = ',', header = None).values[:, self.individual_idx].astype(int)
    return individuals 

  def fit(self):
    # Load data
    train_fps = self.config['train_data_fp']
    train_data = [self.load_model_inputs(fp) for fp in train_fps]
    train_labels = [self.load_labels(fp) for fp in train_fps]
    train_individuals = [self.load_individuals(fp) for fp in train_fps]
    # Create dataloader
    train_dataset = Features(train_data, train_labels, train_individuals, True, self.temporal_window_samples, self.config)
    # Classic models only use labeled points
    if self.balance_classes:
        indices_to_keep = train_dataset.balance_classes_by_individual()
    else:
        indices_to_keep = train_dataset.get_annotated_windows()
        if self.downsizing_factor > 1:
            indices_to_keep = indices_to_keep[::self.downsizing_factor]
    train_dataset = Subset(train_dataset, indices_to_keep)  
    print("Number windowed train examples after subselecting: %d" % len(train_dataset))
    train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=False, num_workers = self.num_workers)
    # Load in all data for fitting     
    X, y = zip(*[(X, y) for X, y in train_dataloader])
    X = torch.cat(X, dim=0).numpy(); y = torch.cat(y).numpy()
    # Fit model
    self.model.fit(X, y)

  def save(self):
    target_fp = os.path.join(self.config['final_model_dir'], "final_model.pickle")
    with open(target_fp, 'wb') as f:
      pickle.dump(self, f)
  
  def predict(self, data):
    dataset = Features([data], None, None, False, self.temporal_window_samples, self.config)
    dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, drop_last=False, num_workers=self.num_workers)
    X = torch.cat([X for X in dataloader], dim=0).numpy()
    return self.model.predict(X), None