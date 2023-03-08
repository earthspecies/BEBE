from sklearn.ensemble import RandomForestClassifier
from scipy import stats
from BEBE.models.model_superclass import BehaviorModel
import yaml
import numpy as np
import pickle
import os
import tqdm
import pandas as pd

def vectorized_slope(y):
  # compute slope along axis = 1, up to a constant multiple which depends only on the length of the time series
  # reduces [batch_size, n_samples_per_batch, features] to [batch_size, features]
  mu_y = np.mean(y, axis = 1, keepdims = True)
  x = np.arange(0, np.shape(y)[1])
  x = np.reshape(x, (1, -1, 1))
  mu_x = np.mean(x)
  return np.sum((x - mu_x) * (y- mu_y) , axis = 1)
  
def autocorr(y, mu, sigma):
    # approximates the 1-sample autocorrelation using previously computed values
    # operates on axis = 1
    # reduces [batch_size, n_samples_per_batch, features] to [batch_size, features]
    mu = np.expand_dims(mu, axis = 1)
    numerator = (y[:, 1:, :] - mu) * (y[:, :-1, :] - mu)
    numerator = np.mean(numerator, axis = 1)
    return numerator / (sigma ** 2 + 1e-12)
  
class rf(BehaviorModel):
  def __init__(self, config):
    super(rf, self).__init__(config)
    
    self.n_samples_window = max(int(np.ceil(self.model_config['context_window_sec'] * self.metadata['sr'])), 3)
    self.min_samples_split = self.model_config['min_samples_split']
    self.max_samples = self.model_config['max_samples']
    self.n_jobs = self.model_config['n_jobs']
    self.n_estimators = self.model_config['n_estimators']
    
    self.model = RandomForestClassifier(n_estimators=self.n_estimators, min_samples_split=self.min_samples_split, n_jobs=self.n_jobs, verbose=2, max_samples=self.max_samples, random_state = self.config['seed'])
    
    labels_bool = [x == 'label' for x in self.metadata['clip_column_names']]
    self.label_idx = [i for i, x in enumerate(labels_bool) if x][0] # int
    
  def load_model_inputs(self, filepath):

    inputs = pd.read_csv(filepath, delimiter = ',', header = None).values
    data = inputs[:, self.cols_included]
      
    return data
  
  def load_labels(self, filepath):
    inputs = pd.read_csv(filepath, delimiter = ',', header = None).values
    labels = inputs[:, self.label_idx].astype(int)

    return labels
  
  def prepare_model_inputs(self, data, labels = None):
    # in: data [seq_len, n_features], labels [seq_len,]
    # out: mask of features [ceil(seq_len / n_samples_window), 8*n_features], labels [ceil(seq_len / n_samples_window),]
    
    # Pad and reshape
    # pad seq_len by adding zeros to the end
    # data [seq_len, n_features] -> [seq_len/n_samples_window, n_samples_window, n_features]
    # labels [seq_len,] -> [seq_len/n_samples_window, n_samples_window] 
    seq_len = np.shape(data)[0]
    pad_left = (self.n_samples_window - 1) // 2
    pad_right = self.n_samples_window - 1 - pad_left
    padded_data = np.pad(data, ((pad_left, pad_right), (0,0)))
    context_data = []

    for i in range(0, self.n_samples_window):
        context_data.append(padded_data[i: i + seq_len, :])
        
    context_data = np.stack(context_data, axis = 1) #[seq_len, context_window_samples, n_features]
    
    # Compute summary statistics (context features)
    features = []
    features.append(np.amax(context_data, axis = 1)) # max value
    features.append(np.amin(context_data, axis = 1)) # min value
    mu = np.mean(context_data, axis = 1)
    features.append(mu) # mean
    sigma = np.std(context_data, axis = 1)
    features.append(np.std(context_data, axis = 1)) # std
    features.append(stats.skew(context_data, axis = 1)) # skew
    features.append(stats.kurtosis(context_data, axis = 1)) # kurtosis
    features.append(vectorized_slope(context_data)) # slope (up to a constant)
    features.append(autocorr(context_data, mu, sigma)) # approximate autocorrelation
    features = np.concatenate([data, *features], axis = -1)
    
    # mask windows where labels is unknown
    if labels is not None:
      mask = labels != 0
      labels = labels[mask]
      features = features[mask]
    else:
      labels = None
      
    return features, labels
    
  def fit(self):
    ## get data. assume stored in memory for now
    train_fps = self.config['train_data_fp']
    
    train_data = []
    train_labels = []
    print("Preparing model inputs")
    for fp in tqdm.tqdm(train_fps):
      data = self.load_model_inputs(fp)
      labels = self.load_labels(fp)
      data, labels = self.prepare_model_inputs(data, labels = labels)
      train_data.append(data)
      train_labels.append(labels)
    
    train_data = np.concatenate(train_data, axis = 0)
    train_labels = np.concatenate(train_labels, axis = 0)
    self.model.fit(train_data, train_labels)
    
  def save(self):
    target_fp = os.path.join(self.config['final_model_dir'], "final_model.pickle")
    with open(target_fp, 'wb') as f:
      pickle.dump(self, f)

  def predict(self, data):
    raw_data_size = np.shape(data)[0]
    data, _ = self.prepare_model_inputs(data)
    predictions = self.model.predict(data)
    return predictions, None
