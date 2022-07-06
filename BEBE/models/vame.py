import yaml
import numpy as np
import pickle
import os
import shutil
import BEBE.applications.VAME.vame as VAME
from BEBE.models.model_superclass import BehaviorModel
from BEBE.models.whiten import whitener_standalone
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
    
    self.whiten = self.model_config['whiten']
    self.whitener = whitener_standalone()
    
    with open(self.config_vame_fp, 'w') as file:
      yaml.dump(config_vame, file)
      
    self.temp_data_dir = os.path.join(self.vame_experiment_dir, 'data', 'train')
    if not os.path.exists(self.temp_data_dir):
      os.makedirs(self.temp_data_dir)
    
  def fit(self):
    ## get data. assume stored in memory for now
    if self.read_latents:
      dev_fps = self.config['dev_data_latents_fp']
      test_fps = self.config['test_data_latents_fp']
    else:
      dev_fps = self.config['dev_data_fp']
      test_fps = self.config['test_data_fp']
      
    # Save off temp files
    dev_data = [self.load_model_inputs(fp, read_latents = self.read_latents) for fp in dev_fps]
    dev_data = np.concatenate(dev_data, axis = 0)
    if self.whiten:
      dev_data = self.whitener.fit_transform(dev_data)
    
    test_data = [self.load_model_inputs(fp, read_latents = self.read_latents) for fp in test_fps]
    test_data = np.concatenate(test_data, axis = 0)
    if self.whiten:
      test_data = self.whitener.transform(test_data)
    
    temp_dev_fp = os.path.join(self.temp_data_dir, 'train_seq.npy')
    np.save(temp_dev_fp, dev_data)
    
    temp_test_fp = os.path.join(self.temp_data_dir, 'test_seq.npy')
    #np.save(temp_test_fp, test_data)
    # We will use dev data for model selection
    np.save(temp_test_fp, dev_data)
    
    # Modify config_vame as necessary
    num_features_vame = np.shape(dev_data)[1] + 2
    
    with open(self.config_vame_fp) as file:
      config_vame = yaml.load(file, Loader=yaml.FullLoader)
      
    config_vame['num_features'] = num_features_vame
    
    with open(self.config_vame_fp, 'w') as file:
      yaml.dump(config_vame, file)
    
    # Free memory
    del dev_data
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
    #target_fp = os.path.join(self.config['final_model_dir'], "final_model.pickle")
    #with open(target_fp, 'wb') as f:
    #  pickle.dump(self, f)
  
  def predict(self, data):
    # not implemented
    # use method predict_from_file instead
    raise NotImplementedError
  
  def predict_from_file(self, fp):
    file_id = fp.split('/')[-1].split('.')[0]    
    inputs = self.load_model_inputs(fp, read_latents = self.read_latents)
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
