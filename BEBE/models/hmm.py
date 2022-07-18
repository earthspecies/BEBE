from BEBE.applications.ssm import ssm
import yaml
import numpy as np
import pickle
import os
from matplotlib import pyplot as plt
from BEBE.models.model_superclass import BehaviorModel

class hmm(BehaviorModel):
  def __init__(self, config):
    super(hmm, self).__init__(config)
    
    self.time_bins = self.model_config['time_bins']
    
  def fit(self):
    if self.read_latents:
      dev_fps = self.config['dev_data_latents_fp']
    else:
      dev_fps = self.config['dev_data_fp']
    
    #######
  
    dev_data = []
    
    rng = np.random.default_rng(seed = 31)
    for fp in dev_fps:
      obs = self.load_model_inputs(fp, read_latents = self.read_latents)
      
      # Possibly subselect from training data:
      # Used to speed up initial hyperparameter tuning
      if self.model_config['subselect_proportion'] < 1.:
        total_len = np.shape(obs)[0]
        to_select = int(np.ceil(total_len * self.model_config['subselect_proportion']))
        start = int(rng.integers(0, high=total_len - to_select))
        obs = obs[start:start + to_select, ...]
        
      obs_list = [obs[i* self.time_bins: (i+1)* self.time_bins, :] for i in range(len(obs)//self.time_bins)]
      dev_data.extend(obs_list)
      
    self.obs_dim = np.shape(dev_data[0])[1]
    
    ###
    # Compute transition matrix prior probabilities
    
    prior_wait_sec = self.model_config['prior_wait_sec']
    prior_wait_samples = int(self.metadata['sr'] * prior_wait_sec)
    prior_diagonal_entry = float(prior_wait_samples) / (1. + prior_wait_samples) # prior probability P_ii
    
    # Compute kappa and alpha based on prior diagonal entry
    
    prior_kappa_unscaled = (self.config['num_clusters'] * prior_diagonal_entry - 1.) / (self.config['num_clusters'] -1 )
    prior_alpha_unscaled = (1. - prior_kappa_unscaled) / self.config['num_clusters']
    
    assert np.isclose(prior_kappa_unscaled + self.config['num_clusters'] * prior_alpha_unscaled, 1.)
    
    # Because of how ssm handles sticky transitions, 
    # we have to specify how flat is the prior distribution over P_ii
    
    prior_strength = self.model_config['sticky_prior_strength'] # 0 is non-informative prior, 1. is as if we already have num_transitions_train_seen-many samples of evidence for the prior 
    num_transitions_train_seen = len(dev_data) * (self.time_bins-1)
    
    prior_scaling_factor = prior_strength * num_transitions_train_seen
    
    prior_kappa = prior_kappa_unscaled * prior_scaling_factor
    prior_alpha = prior_alpha_unscaled * prior_scaling_factor + 1. # account for shift by 1. in sticky transition
    
    ###
    # Instantiate HMM model
    
    if self.model_config['lags'] > 0:
      self.model = ssm.HMM(self.config['num_clusters'],
                           self.obs_dim,
                           observations= "no_input_ar",
                           observation_kwargs = {"lags" : self.model_config['lags']},
                           transitions = "sticky", 
                           transition_kwargs = {"alpha" : prior_alpha, "kappa" : prior_kappa}, 
                          )
    else:
      self.model = ssm.HMM(self.config['num_clusters'],
                           self.obs_dim,
                           observations= "gaussian",
                           transitions = "sticky", 
                           transition_kwargs = {"alpha" : prior_alpha, "kappa" : prior_kappa}, 
                          )
      
    N_iters = self.model_config['N_iters']
    hmm_lls = self.model.fit(dev_data, 
                             method= "em", 
                             num_iters = N_iters, 
                             # num_epochs = N_iters,
                             init_method="kmeans",
                             # step_size=0.0001
                            )
    
    # Save log likelihood and transition plots
    plt.plot(hmm_lls, label="EM")
    plt.title("Train log probability")
    plt.xlabel("EM Iteration")
    plt.ylabel("Log Probability")
    lls_fp = os.path.join(self.config['visualization_dir'], 'train_lls.png')
    plt.savefig(lls_fp)
    plt.close()
    
    trans_fp = os.path.join(self.config['visualization_dir'], 'trans_probs.png')
    plt.imshow(self.model.transitions.transition_matrix)
    plt.savefig(trans_fp)
    plt.close()
    #######
    
  def save(self):
    target_fp = os.path.join(self.config['final_model_dir'], "final_model.pickle")
    with open(target_fp, 'wb') as f:
      pickle.dump(self, f)
  
  def predict(self, data):
    predictions = self.model.most_likely_states(data)
    return predictions, None
