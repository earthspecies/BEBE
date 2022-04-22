import torch
from torch.utils.data.dataset import Dataset
import numpy as np
import os
import tqdm

class CPCDataset(Dataset):
    def __init__(self, data, train, temporal_window_samples, individual_idx = -2):
        self.temporal_window = temporal_window_samples
        self.individual_idx = individual_idx
        
        self.data = data # list of np arrays, each of shape [*, n_features] where * is the number of samples and varies between arrays
        
        self.data_points = sum([np.shape(x)[0] for x in self.data])
        
        print('Initialize dataloader. Datapoints %d' %self.data_points)
            
        self.data_start_indices = []
        counter = 0
        for x in self.data:
          assert np.shape(x)[0] > temporal_window_samples, "temporal_window_samples must be shorter than smallest example"
          self.data_start_indices.append(counter)
          counter = counter + np.shape(x)[0] - self.temporal_window
          
        assert counter == self.data_points - len(self.data) * self.temporal_window
        self.data_start_indices = np.array(self.data_start_indices)
        
        self.data_stds = np.std(np.concatenate(self.data, axis = 0), axis = 0, keepdims = True) / 8
        self.num_channels = np.shape(self.data_stds)[1]
        self.rng = np.random.default_rng()
        self.train = train
        
    def __len__(self):        
        return self.data_points - len(self.data) * self.temporal_window

    def __getitem__(self, index):
        clip_number = np.where(index >= self.data_start_indices)[0][-1] #which clip do I draw from?
        
        data_item = self.data[clip_number]
        
        start = index - self.data_start_indices[clip_number]
        end = start+ self.temporal_window     
        
        individual_id = data_item[0, self.individual_idx]
        data_item = data_item[start:end, :-2]
        
        
        ind_id_one_hot = np.zeros((31,))
        ind_id_one_hot[int(individual_id)] = 1.        
        
        return torch.from_numpy(data_item), torch.from_numpy(ind_id_one_hot)
