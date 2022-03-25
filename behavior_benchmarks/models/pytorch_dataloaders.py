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
        
        if train:
            print('Initialize train data. Datapoints %d' %self.data_points)
        else:
            print('Initialize test data. Datapoints %d' %self.data_points)
            
        self.data_start_indices = []
        counter = 0
        for x in self.data:
          assert np.shape(x)[0] > temporal_window_samples, "temporal_window_samples must be shorter than smallest train example"
          self.data_start_indices.append(counter)
          counter = counter + np.shape(x)[0] - self.temporal_window
          
        assert counter == self.data_points - len(self.data) * self.temporal_window
        self.data_start_indices = np.array(self.data_start_indices)
        
        #self.data = np.concatenate(self.data, axis = 0)
        #self.labels = np.concatenate(self.labels, axis = 0)
        
            
        
    def __len__(self):        
        return self.data_points - len(self.data) * self.temporal_window

    def __getitem__(self, index):
      
        
        clip_number = np.where(index >= self.data_start_indices)[0][-1] #which clip do I draw from?
        start = index - self.data_start_indices[clip_number]
        
        
      
        #start = min(index, self.data_points - self.temporal_window)   #Treat last temporal_window elements as the same.
        end = start+ self.temporal_window
        
        data_item = self.data[clip_number][start:end, :]
        labels_item = self.labels[clip_number][start:end] #[:, start:end]
            
        return torch.from_numpy(data_item), torch.from_numpy(labels_item)
    
    
    
    
    
