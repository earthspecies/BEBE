import numpy as np
import jax
import jax.random as jr
import jax.numpy as jnp
import pandas as pd
import pickle
import os
from matplotlib import pyplot as plt

from dynamax.hidden_markov_model import GaussianHMM, DiagonalGaussianHMM

from BEBE.models.preprocess import static_acc_filter, load_wavelet_transformed_data
from BEBE.models.model_superclass import BehaviorModel
import torch


class hmm(BehaviorModel):
  def __init__(self, config):
    super(hmm, self).__init__(config)
    self.random_seed = self.config['seed']
    self.oom_limit = 10000

    ## Model setup
    if self.model_config["covariance"] == "full":
      self.model_creation = lambda emission_dim: GaussianHMM(
          self.config['num_clusters'], 
          emission_dim, 
          initial_probs_concentration=1.1, 
          transition_matrix_concentration=self.model_config["matrix_concentration"],
          transition_matrix_stickiness=0.0, 
          emission_prior_mean=0.0, 
          emission_prior_concentration=self.model_config["mean_concentration"], 
          emission_prior_scale=self.model_config["cov_scale"],
          emission_prior_extra_df=self.model_config["cov_df"]
        )
    elif self.model_config["covariance"] == "diagonal":
      self.model_creation = lambda emission_dim: DiagonalGaussianHMM(
                          self.config['num_clusters'], 
                          emission_dim, 
                          initial_probs_concentration=1.1, 
                          transition_matrix_concentration=self.model_config["matrix_concentration"], 
                          transition_matrix_stickiness=0.0, 
                          emission_prior_mean=0.0, 
                          emission_prior_mean_concentration=self.model_config["mean_concentration"], 
                          emission_prior_concentration=self.model_config["cov_concentration"], 
                          emission_prior_scale=self.model_config["cov_scale"]
                          )

    ## Pre-processing setup
    self.wavelet_transform = self.model_config['wavelet_transform']
    # wavelet transform: specific settings
    self.morlet_w = self.model_config['morlet_w']
    self.n_wavelets = self.model_config['n_wavelets']
    self.downsample = self.model_config['downsample']

  def load_model_inputs(self, filepath, train=False):
    if not self.wavelet_transform:
      data = pd.read_csv(filepath, delimiter = ',', header = None).values[:, self.cols_included]
      data = static_acc_filter(data, self.config)
    else:
      data = load_wavelet_transformed_data(self, filepath, downsample = self.downsample)
    if train: #for batching purposes
      if data.shape[0] % self.model_config["temporal_window_samples"] != 0:
        data = data[:-(data.shape[0] % self.model_config["temporal_window_samples"]), :]
    return data

  def zscore(self, data, train=False):
    if train:
      self.mu = data.reshape(-1, data.shape[-1]).mean(0)
      self.std = data.reshape(-1, data.shape[-1]).std(0)
    if len(data.shape) == 2: #time x features
      return (data - self.mu[None, :])/self.std[None, :]
    elif len(data.shape) == 3: #batch x time x features
      return (data - self.mu[None, None, :])/self.std[None, None, :]

  def fit(self):

    train_data = []
    for fp in self.config["train_data_fp"]:
        x = self.load_model_inputs(fp, train=True)
        x_batches = np.stack(np.split(x, int(x.shape[0]/self.model_config["temporal_window_samples"]), axis=0))
        train_data.append(x_batches)

    train_data = np.concatenate(train_data, axis=0)
    train_data = self.zscore(train_data, train=True)
    n_batches, self.obs_dim, self.feature_dim = train_data.shape
    ###
    self.jax_device = "cpu" if ( self.model_config["covariance"] == "full" or (not torch.cuda.is_available()) ) else "gpu"
    with jax.default_device(jax.devices(self.jax_device)[0]):
      key = jr.PRNGKey(self.config['seed'])
      train_data = jnp.array(train_data)
      self.model = self.model_creation(self.feature_dim)
      params, param_props = self.model.initialize(key=key, method="kmeans", emissions=train_data)
      self.model_params, log_probs = self.model.fit_em(params, param_props, train_data, num_iters=self.model_config["N_iters"])
      bad_optimization = np.any(np.isnan(log_probs))

    # Save log likelihood and transition plots
    plt.plot(log_probs, label="EM")
    plt.title(f"Train log probability, trained to completion: {not bad_optimization}")
    plt.xlabel("EM Iteration")
    plt.ylabel("Log Probability")
    lls_fp = os.path.join(self.config['visualization_dir'], 'train_lls.png')
    plt.savefig(lls_fp)
    plt.close()

    if bad_optimization:
      raise Exception(f"HMM did not optimize for all {self.model_config['N_iters']} iterations, ran into NaNs.")

    trans_fp = os.path.join(self.config['visualization_dir'], 'trans_probs.png')
    plt.imshow(self.model_params.transitions.transition_matrix)
    plt.savefig(trans_fp)
    plt.close()
    
  def save(self):
    self.model_creation = None #Pickle can't save local lambda, and we don't need it anymore
    target_fp = os.path.join(self.config['final_model_dir'], "final_model.pickle")
    with open(target_fp, 'wb') as f:
      pickle.dump(self, f)
  
  def predict(self, data):
    # assert len(data.shape) == 2 #time, features
    data = self.zscore(data, train=False)
    with jax.default_device(jax.devices(self.jax_device)[0]):
      data = jnp.array(data)
      predictions = self.model.most_likely_states(self.model_params, data)
    return predictions, None