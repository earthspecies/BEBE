from behavior_benchmarks.applications.ssm import ssm
import yaml
import numpy as np
import pickle
import os
from matplotlib import pyplot as plt

class hmm():
  def __init__(self, config):
    self.config = config
    self.read_latents = config['read_latents']
    self.model_config = config['hmm_config']
    
    self.metadata = config['metadata']
      
    cols_included_bool = [x in self.config['input_vars'] for x in self.metadata['clip_column_names']] 
    self.cols_included = [i for i, x in enumerate(cols_included_bool) if x]
    
    self.time_bins = self.model_config['time_bins']
    #self.obs_dim = len(self.config['input_vars'])
  
  def load_model_inputs(self, filepath, read_latents = False):
    if read_latents:
      return np.load(filepath)
    else:
      return np.load(filepath)[:, self.cols_included]
    
  def fit(self):
    ## get data. assume stored in memory for now
    if self.read_latents:
      train_fps = self.config['train_data_latents_fp']
    else:
      train_fps = self.config['train_data_fp']
    
    #######
  
    train_data = []
    for fp in train_fps:
      obs = self.load_model_inputs(fp, read_latents = self.read_latents)
      obs_list = [obs[i* self.time_bins: (i+1)* self.time_bins, :] for i in range(len(obs)//self.time_bins)]
      train_data.extend(obs_list)
      
    self.obs_dim = np.shape(train_data[0])[1]
    
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
    # This is a bit strange how it's written
    
    prior_strength = self.model_config['sticky_prior_strength'] # 0 is non-informative prior, 1. is as if we already have num_transitions_train_seen-many samples of evidence for the prior 
    num_transitions_train_seen = len(train_data) * (self.time_bins-1)
    
    prior_scaling_factor = prior_strength * num_transitions_train_seen
    
    prior_kappa = prior_kappa_unscaled * prior_scaling_factor
    prior_alpha = prior_alpha_unscaled * prior_scaling_factor + 1. # account for shift by 1. in sticky transition
    
    ###
      
    self.model = ssm.HMM(self.config['num_clusters'],
                         self.obs_dim,
                         observations= "no_input_ar",
                         observation_kwargs = {"lags" : self.model_config['lags']},
                         transitions = "sticky", 
                         transition_kwargs = {"alpha" : prior_alpha, "kappa" : prior_kappa}, 
                        )  
      
    N_iters = self.model_config['N_iters']
    hmm_lls = self.model.fit(train_data, 
                             method= "em", 
                             num_iters=N_iters, 
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
  
  
  def predict_from_file(self, fp):
    inputs = self.load_model_inputs(fp, read_latents = self.read_latents)
    predictions, latents = self.predict(inputs)
    return predictions, latents
