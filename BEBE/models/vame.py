import yaml
import numpy as np
import pickle
import os
import shutil
import BEBE.applications.VAME.vame as VAME
from BEBE.models.model_superclass import BehaviorModel
from BEBE.models.preprocess import whitener_standalone
import torch
from matplotlib import pyplot as plt

class vame(BehaviorModel):
  def __init__(self, config):
    super(vame, self).__init__(config)
    
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        print("Using CUDA")
        print('GPU active:',torch.cuda.is_available())
        print('GPU used:',torch.cuda.get_device_name(0))
    
    self.model = None
    
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
    config_vame['max_epochs'] = self.get_n_epochs()
    config_vame['beta'] = self.model_config['beta']
    config_vame['zdims'] = self.model_config['zdims']
    config_vame['learning_rate'] = self.model_config['learning_rate']
    config_vame['time_window'] = int(self.model_config['time_window_sec'] * self.metadata['sr'])
    config_vame['prediction_decoder'] = self.model_config['prediction_decoder']
    config_vame['prediction_steps'] = int(self.model_config['time_window_sec'] * self.metadata['sr']) # use time window = prediction window
    config_vame['scheduler'] = self.model_config['scheduler']
    config_vame['scheduler_step_size'] = self.model_config['scheduler_step_size']
    config_vame['scheduler_gamma'] = self.model_config['scheduler_gamma']
    config_vame['kmeans_loss'] = self.model_config['zdims'] # Uses all singular values
    config_vame['kmeans_lambda'] = self.model_config['kmeans_lambda']
    config_vame['downsizing_factor'] = self.model_config['downsizing_factor']
    config_vame['model_convergence'] = self.model_config['max_epochs'] # Do not use early stopping
    config_vame['seed'] = self.config['seed']
    
    self.whiten = self.model_config['whiten']
    self.whitener = whitener_standalone()
    
    with open(self.config_vame_fp, 'w') as file:
      yaml.dump(config_vame, file)
      
    self.temp_data_dir = os.path.join(self.vame_experiment_dir, 'data', 'train')
    if not os.path.exists(self.temp_data_dir):
      os.makedirs(self.temp_data_dir)
      
  def get_n_epochs(self):
    train_fps = self.config['train_data_fp']
    train_data = [self.load_model_inputs(fp) for fp in train_fps]
    train_data = np.concatenate(train_data, axis = 0)
    data_len = np.shape(train_data)
    max_n_epochs = int(np.ceil((self.model_config['n_train_steps'] * self.model_config['batch_size']) / data_len))
    return max_n_epochs
    
  def fit(self):
    ## get data. assume stored in memory for now
    train_fps = self.config['train_data_fp']
    test_fps = self.config['test_data_fp']
      
    # Save off temp files
    train_data = [self.load_model_inputs(fp) for fp in train_fps]
    train_data = np.concatenate(train_data, axis = 0)
    if self.whiten:
      train_data = self.whitener.fit_transform(train_data)
    
    test_data = [self.load_model_inputs(fp) for fp in test_fps]
    test_data = np.concatenate(test_data, axis = 0)
    if self.whiten:
      test_data = self.whitener.transform(test_data)
    
    temp_train_fp = os.path.join(self.temp_data_dir, 'train_seq.npy')
    np.save(temp_train_fp, train_data)
    
    temp_test_fp = os.path.join(self.temp_data_dir, 'test_seq.npy')
    # We will use dev data for model selection
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
    train_losses, test_losses, kmeans_losses, kl_losses, weight_values, mse_losses, fut_losses = VAME.train_model(self.config_vame_fp)
    plt.plot(train_losses, label = 'train_loss')
    plt.plot(test_losses, label = 'test_loss')
    plt.plot(kmeans_losses, label = 'train_kmeans_loss')
    plt.plot(weight_values, label = 'weight_value')
    plt.plot(mse_losses, label = 'train_mse_loss')
    plt.plot(fut_losses, label = 'train_future_loss')
    plt.legend()
    plt.savefig(os.path.join(self.config['visualization_dir'], 'training_progress.png'))
    plt.close()
    
  def save(self):
    if os.path.exists(self.config['final_model_dir']):
      shutil.rmtree(self.config['final_model_dir'])    
    shutil.copytree(os.path.join(self.vame_experiment_dir, 'model'), self.config['final_model_dir'])
  
  def predict(self, data):
    # not implemented
    # use method predict_from_file instead
    raise NotImplementedError
  
  def predict_from_file(self, fp):
    file_id = fp.split('/')[-1].split('.')[0]    
    inputs = self.load_model_inputs(fp)
    if self.whiten:
      inputs = self.whitener.transform(inputs)
    
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
