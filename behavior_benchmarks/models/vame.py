import yaml
import numpy as np
import pickle
import os
import shutil
import behavior_benchmarks.applications.VAME.vame as VAME

class vame():
  def __init__(self, config):
    self.config = config
    self.read_latents = config['read_latents']
    self.model_config = config['vame_config']
    self.model = None
    self.metadata = config['metadata']
      
    cols_included_bool = [x in self.config['input_vars'] for x in self.metadata['clip_column_names']] 
    self.cols_included = [i for i, x in enumerate(cols_included_bool) if x]
    
    # Set up temporary VAME directory to follow the VAME package's formatting conventions
    project = 'temp_vame_project'
    self.config_vame_fp = VAME.init_new_project(project=project, videos=[], working_directory=config['temp_dir'])
    
    # Modify config_vame to reflect our conventions
    with open(self.config_vame_fp) as file:
      config_vame = yaml.load(file, Loader=yaml.FullLoader)
      
    self.vame_experiment_dir = config_vame['project_path']
  
    config_vame['n_cluster'] = config['num_clusters']
    config_vame['n_init_kmeans'] = config['num_clusters']
    config_vame['batch_size'] = self.model_config['batch_size']
    config_vame['max_epochs'] = self.model_config['max_epochs']
    config_vame['beta'] = self.model_config['beta']
    config_vame['zdims'] = self.model_config['zdims']
    config_vame['learning_rate'] = self.model_config['learning_rate']
    config_vame['time_window'] = int(self.model_config['time_window_sec'] * self.metadata['sr'])
    config_vame['prediction_decoder'] = self.model_config['prediction_decoder']
    config_vame['prediction_steps'] = int(self.model_config['prediction_sec'] * self.metadata['sr'])
    config_vame['scheduler'] = self.model_config['scheduler']
    config_vame['scheduler_step_size'] = self.model_config['scheduler_step_size']
    config_vame['scheduler_gamma'] = self.model_config['scheduler_gamma']
    config_vame['kmeans_loss'] = self.model_config['zdims'] # Uses all singular values
    config_vame['kmeans_lambda'] = self.model_config['kmeans_lambda']
    config_vame['downsizing_factor'] = self.model_config['downsizing_factor']
    
    with open(self.config_vame_fp, 'w') as file:
      yaml.dump(config_vame, file)
      
    self.temp_data_dir = os.path.join(self.vame_experiment_dir, 'data', 'train')
    if not os.path.exists(self.temp_data_dir):
      os.makedirs(self.temp_data_dir)
    
  
  def load_model_inputs(self, filepath, read_latents = False):
    if read_latents:
      return np.load(filepath)
    else:
      return np.load(filepath)[:, self.cols_included]
    
  def fit(self):
    ## get data. assume stored in memory for now
    if self.read_latents:
      train_fps = self.config['train_data_latents_fp']
      test_fps = self.config['test_data_latents_fp']
    else:
      train_fps = self.config['train_data_fp']
      test_fps = self.config['test_data_fp']
      
    # Save off temp files
    train_data = [self.load_model_inputs(fp, read_latents = self.read_latents) for fp in train_fps]
    train_data = np.concatenate(train_data, axis = 0)
    
    test_data = [self.load_model_inputs(fp, read_latents = self.read_latents) for fp in test_fps]
    test_data = np.concatenate(test_data, axis = 0)
    
    temp_train_fp = os.path.join(self.temp_data_dir, 'train_seq.npy')
    np.save(temp_train_fp, train_data)
    
    temp_test_fp = os.path.join(self.temp_data_dir, 'test_seq.npy')
    #np.save(temp_test_fp, test_data)
    # We will use train data for model selection, to not affect validity of test metrics
    # Better would be to have a explicit val set
    np.save(temp_test_fp, train_data)
    
    # Modify config_vame as necessary
    num_features_vame = np.shape(train_data)[1] + 2
    
    with open(self.config_vame_fp) as file:
      config_vame = yaml.load(file, Loader=yaml.FullLoader)
      
    config_vame['num_features'] = num_features_vame
    
    with open(self.config_vame_fp, 'w') as file:
      yaml.dump(config_vame, file)
    
    # Free memory
    del train_data
    del test_data
    
    # Train
    VAME.train_model(self.config_vame_fp)
    
  def save(self):
    if os.path.exists(self.config['final_model_dir']):
      shutil.rmtree(self.config['final_model_dir'])    
    shutil.copytree(os.path.join(self.vame_experiment_dir, 'model'), self.config['final_model_dir'])
    #target_fp = os.path.join(self.config['final_model_dir'], "final_model.pickle")
    #with open(target_fp, 'wb') as f:
    #  pickle.dump(self, f)
  
  def predict(self, data):
    # not implemented
    raise ValueError
  
  def predict_from_file(self, fp):
    file_id = fp.split('/')[-1].split('.')[0]    
    inputs = self.load_model_inputs(fp, read_latents = self.read_latents)
    
    # Save temporary version of data for VAME to see
    temp_fp = os.path.join(self.temp_data_dir, file_id + '_seq.npy')
    np.save(temp_fp, inputs)
    
    # Modify config to reflect this
    with open(self.config_vame_fp) as file:
      config_vame = yaml.load(file, Loader=yaml.FullLoader)  
    config_vame['video_sets'] = [file_id]
    with open(self.config_vame_fp, 'w') as file:
      yaml.dump(config_vame, file)
    
    # Operate the VAME model
    VAME.pose_segmentation(self.config_vame_fp)
    temp_results_fp = os.path.join(self.vame_experiment_dir,"results","")
    
    # Load up what it saved off
    predictions_fp = os.path.join(temp_results_fp,'km_label_'+file_id + '.npy')
    predictions = np.load(predictions_fp)
    
    latents_fp = os.path.join(temp_results_fp,'latent_vector_'+file_id + '.npy')
    latents = np.load(latents_fp)
                              
    return predictions, latents
