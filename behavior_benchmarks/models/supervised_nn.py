import yaml
import numpy as np
import pickle
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from behavior_benchmarks.models.pytorch_dataloaders import BEHAVIOR_DATASET

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

class supervised_nn():
  def __init__(self, config):
    self.config = config
    self.read_latents = config['read_latents']
    self.model_config = config['supervised_nn_config']
    self.metadata = config['metadata']
    
    cols_included_bool = [x in self.config['input_vars'] for x in self.metadata['clip_column_names']] 
    self.cols_included = [i for i, x in enumerate(cols_included_bool) if x]
    
    labels_bool = [x == 'label' for x in self.metadata['clip_column_names']]
    self.label_idx = [i for i, x in enumerate(labels_bool) if x][0] # int
    
    self.n_classes = len(self.metadata['label_names']) 
    self.n_features = len(self.cols_included)
    
    self.model = NeuralNetwork(self.n_features, self.n_classes).to(device)
    print(self.model)
  
  def load_model_inputs(self, filepath, read_latents = False):
    if read_latents:
      raise NotImplementedError("Supervised model is expected to read from raw data")
      #return np.load(filepath)
    else:
      
      return np.load(filepath)[:, self.cols_included].T #[n_features, n_samples]
    
  def load_labels(self, filepath):
    labels = np.load(filepath)[:, self.label_idx].astype(int)
    one_hot_labels = np.eye(self.n_classes)[:, labels]
    return labels #one_hot_labels # one-hot encoded of shape [n_classes, n_samples]
    
  def fit(self):
    ## get data. assume stored in memory for now
    if self.read_latents:
      train_fps = self.config['train_data_latents_fp']
    else:
      train_fps = self.config['train_data_fp']
    
    train_data = [self.load_model_inputs(fp, read_latents = self.read_latents) for fp in train_fps]
    train_data = np.concatenate(train_data, axis = 1)
    
    train_labels = [self.load_labels(fp) for fp in train_fps]
    train_labels = np.concatenate(train_labels, axis = 0)
    
    ## TODO: include idx for file number so we don't sample from multiple files
    
    train_dataset = BEHAVIOR_DATASET(train_data, train_labels, True, 80)
    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True, drop_last=True, num_workers = 8)
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-3)
    
    epochs = 5
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, self.model, loss_fn, optimizer)
        #test(test_dataloader, model, loss_fn)
    print("Done!")
    ###
    
  def save(self):
    target_fp = os.path.join(self.config['final_model_dir'], "final_model.pickle")
    with open(target_fp, 'wb') as f:
      pickle.dump(self, f)
  
  def predict(self, data):
    ###
    predictions = self.model.predict(data)
    return predictions, None
    ###
  
  def predict_from_file(self, fp):
    ###
    inputs = self.load_model_inputs(fp, read_latents = self.read_latents)
    predictions, latents = self.predict(inputs)
    return predictions, latents
    ###


class NeuralNetwork(nn.Module):
    def __init__(self, n_features, n_classes):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.conv_relu_stack = nn.Sequential(
            nn.Conv1d(n_features, 8, 5, padding = 'same'),
            nn.ReLU(),
            nn.Conv1d(8, 8, 5, padding = 'same'),
            nn.ReLU(),
            nn.Conv1d(8, n_classes, 5, padding = 'same') 
        )

    def forward(self, x):
        #x = self.flatten(x)
        logits = self.conv_relu_stack(x)
        return logits

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.type('torch.FloatTensor').to(device), y.type('torch.LongTensor').to(device)

        # Compute prediction error
        pred = model(X)
        
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")