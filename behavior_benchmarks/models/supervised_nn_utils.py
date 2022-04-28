import torch
from torch.utils.data.dataset import Dataset
import numpy as np
import os
import tqdm
    
class BEHAVIOR_DATASET(Dataset):
    def __init__(self, data, labels, train, temporal_window_samples, rescale_param = 0):
        self.temporal_window = temporal_window_samples
        self.rescale_param = rescale_param
        
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
      
    def get_annotated_windows(self):
        # Go through data, make a list of the indices of windows which actually have annotations.
        # Useful for speeding up supervised train time for sparsely labeled datasets.
        
        indices_of_annotated_windows = []
        print("Subselecting data to speed up training")
        for index in tqdm.tqdm(range(self.__len__())):
          clip_number = np.where(index >= self.data_start_indices)[0][-1] #which clip do I draw from?
          labels_item = self.labels[clip_number]
        
          #start = min(index, self.data_points - self.temporal_window)   #Treat last temporal_window elements as the same.
          start = index - self.data_start_indices[clip_number]
          end = start+ self.temporal_window
          
          if np.any(labels_item[start:end] != 0):
            indices_of_annotated_windows.append(index)
        return indices_of_annotated_windows

    def __getitem__(self, index):
        clip_number = np.where(index >= self.data_start_indices)[0][-1] #which clip do I draw from?
        
        data_item = self.data[clip_number]
        labels_item = self.labels[clip_number]
        
        #start = min(index, self.data_points - self.temporal_window)   #Treat last temporal_window elements as the same.
        start = index - self.data_start_indices[clip_number]
        end = start+ self.temporal_window
        
        if self.train and self.rescale_param:
          # rescale augmentation
          max_rescale_size = int(self.rescale_param * self.temporal_window)
          shift = self.rng.integers(-max_rescale_size, max_rescale_size)
          end += shift
          end = min(np.shape(data_item)[0], end)
          samples = np.linspace(start, end, num = self.temporal_window, endpoint = False, dtype = int)
          data_item = data_item[samples, :]
          labels_item = labels_item[samples]
        
        else:
          data_item = data_item[start:end, :]
          labels_item = labels_item[start:end] #[:, start:end]
        
#         if self.train:
#           blur = self.rng.normal(scale = self.data_stds, size = (self.temporal_window, self.num_channels))/8
#           data_item = data_item +  2 * blur[:1, :] + blur
            
        return torch.from_numpy(data_item), torch.from_numpy(labels_item)
