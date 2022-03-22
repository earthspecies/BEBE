import torch
from torch.utils.data.dataset import Dataset
import numpy as np
import os

class BEHAVIOR_DATASET(Dataset):
    def __init__(self, data, labels, train, temporal_window_samples):
        self.temporal_window = temporal_window_samples
        self.data = data
        self.labels = labels
        
        self.data_points = np.shape(self.data)[1]
        
        if train:
            print('Initialize train data. Datapoints %d' %self.data_points)
        else:
            print('Initialize test data. Datapoints %d' %self.data_points)
        
    def __len__(self):        
        return self.data_points

    def __getitem__(self, index):
        start = min(index, self.data_points - self.temporal_window)   #Treat last temporal_window elements as the same.
        end = start+ self.temporal_window
        
        data_item = self.data[:, start:end]
        labels_item = self.labels[start:end] #[:, start:end]
            
        return torch.from_numpy(data_item), torch.from_numpy(labels_item)
    
    
    
    
    
