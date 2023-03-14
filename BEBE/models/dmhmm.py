import numpy as np
import pickle
import os
from matplotlib import pyplot as plt
from BEBE.models.model_superclass import BehaviorModel
from dynamax.hidden_markov_model import GaussianHMM, DiagonalGaussianHMM, LinearAutoregressiveHMM
import jax.random as jr
# from functools import partial
import jax.numpy as jnp
from jax import vmap
import warnings

## Example config
# model_config:
#   time_bins: 1000
#   N_iters: 10
#   lags: 1
#   type: autoregressive
#   matrix_concentration: 1.1

class hmm(BehaviorModel):
  def __init__(self, config):
    super(hmm, self).__init__(config)
    self.random_seed = self.config['seed']
    self.oom_limit = 10000
    if self.model_config["type"] == "full-covariance":
      self.model_creation = lambda emission_dim: GaussianHMM(
          self.config['num_clusters'], 
          emission_dim, 
          initial_probs_concentration=1.1, 
          transition_matrix_concentration=self.model_config["matrix_concentration"], #1.1
          transition_matrix_stickiness=0.0, 
          emission_prior_mean=0.0, 
          emission_prior_concentration=0.1, 
          emission_prior_scale=0.1, 
          emission_prior_extra_df=0.1
        )
    elif self.model_config["type"] == "diagonal-covariance":
      self.model_creation = lambda emission_dim: DiagonalGaussianHMM(
                          self.config['num_clusters'], 
                          emission_dim, 
                          initial_probs_concentration=1.1, 
                          transition_matrix_concentration=self.model_config["matrix_concentration"], 
                          transition_matrix_stickiness=0.0, 
                          emission_prior_mean=0.0, 
                          emission_prior_mean_concentration=0.1, 
                          emission_prior_concentration=0.1, 
                          emission_prior_scale=0.1
                          )
    elif self.model_config["type"] == "autoregressive":
      self.model_creation = lambda emission_dim: LinearAutoregressiveHMM(
                              self.config['num_clusters'], 
                              emission_dim, 
                              num_lags=self.model_config['lags'], 
                              initial_probs_concentration=1.1, 
                              transition_matrix_concentration=self.model_config["matrix_concentration"], 
                              transition_matrix_stickiness=0.0
                              )

  def fit(self):

    train_data_list = []
    for fp in self.config["train_data_fp"]:
      x = self.load_model_inputs(fp)
      x = x[:-(x.shape[0] % self.model_config["time_bins"]), :]
      x_batches = np.stack(np.split(x, int(x.shape[0]/self.model_config["time_bins"]), axis=0))
      train_data_list.append(x_batches)

    train_data = np.concatenate(train_data_list, axis=0)    
    self.obs_dim = np.shape(train_data)[1]
    self.feature_dim = np.shape(train_data)[2]
    
    ###
    key = jr.PRNGKey(self.config['seed'])
    train_data = jnp.array(train_data)
    self.model = self.model_creation(self.feature_dim)
    params, param_props = self.model.initialize(key=key, method="kmeans", emissions=train_data)
    if self.model_config["type"] == "autoregressive":
      #This does not optimize well: NaNs after ~5 iterations.
      inputs = vmap(self.model.compute_inputs)(train_data) #train_data.shape must be batch x obs_dim x feature_dim
      self.model_params, log_probs = self.model.fit_em(params, param_props, train_data, inputs=inputs, num_iters=self.model_config["N_iters"])
    else:
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
    if self.model_config["type"] == "autoregressive":
      inputs = self.model.compute_inputs(data) 
      predictions = self.model.most_likely_states(self.model_params, data, inputs=inputs)
    else:
      predictions = self.model.most_likely_states(self.model_params, data)
    return predictions, None
