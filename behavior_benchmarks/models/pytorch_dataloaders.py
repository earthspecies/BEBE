import torch
from torch.utils.data.dataset import Dataset
import numpy as np
import os
    
class BEHAVIOR_DATASET(Dataset):
    def __init__(self, data, labels, train, temporal_window_samples):
        self.temporal_window = temporal_window_samples
        
        self.data = data # list of np arrays, each of shape [*, n_features] where * is the number of samples and varies between arrays
        self.labels = labels # list of np arrays, each of shape [*,] where * is the number of samples and varies between arrays
        
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
        #self.labels = np.concatenate(self.labels, axis = 0)
        self.rng = np.random.default_rng()
        self.train = train
        
    def __len__(self):        
        return self.data_points - len(self.data) * self.temporal_window

    def __getitem__(self, index):
        clip_number = np.where(index >= self.data_start_indices)[0][-1] #which clip do I draw from?
        start = index - self.data_start_indices[clip_number]
        
        #start = min(index, self.data_points - self.temporal_window)   #Treat last temporal_window elements as the same.
        end = start+ self.temporal_window
        
        data_item = self.data[clip_number][start:end, :]
        
        if self.train:
          blur = self.rng.normal(scale = self.data_stds, size = (self.temporal_window, self.num_channels))
          data_item = data_item +  2 * blur[:1, :] #+ blur
        
        labels_item = self.labels[clip_number][start:end] #[:, start:end]
            
        return torch.from_numpy(data_item), torch.from_numpy(labels_item)
