import yaml
import numpy as np
import pickle
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import tqdm
from behavior_benchmarks.models.model_superclass import BehaviorModel
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"

def _count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class reducer_nn():
  def __init__(self, config, input_dims, output_dims, layers = 3, hidden_size = 64):
    super(reducer_nn, self).__init__()
    print(f"Using {device} device")
    
    self.model = MLP(input_dims, output_dims, layers, hidden_size).to(device)
    self.config = config
    
    # print(self.model)
    print('Parametric model parameters:')
    print(_count_parameters(self.model))
  
  def fit(self, inputs, targets, train_epochs = 50, lr_init = .001, batch_size = 512):
    
    dataset = UMAP_DATASET(inputs, targets)
    
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers = 0)
    
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(self.model.parameters(), lr=lr_init, amsgrad = True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, train_epochs, eta_min=0, last_epoch=- 1, verbose=False)
    
    train_loss = []
    learning_rates = []
    
    
    with tqdm.tqdm(list(range(train_epochs)), unit = "epoch", total = train_epochs) as tepoch:
      for i in tepoch:
        l = self.train_epoch(train_dataloader, loss_fn, optimizer)
        train_loss.append(l)

        learning_rates.append(optimizer.param_groups[0]["lr"])
        scheduler.step()
        loss_str = "%2.2f" % l
        tepoch.set_postfix(loss=loss_str)        
      
    print("Done!")
    
    ## Save training progress
    
    # Loss
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    
    ax.plot(train_loss, label= 'train', marker = '.')
    ax.set_title("MSE Loss")
    ax.set_xlabel('Epoch')
    
    major_tick_spacing = max(1, len(train_loss) // 10)
    ax.xaxis.set_major_locator(MultipleLocator(major_tick_spacing))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.set_ylabel('Loss')
    loss_fp = os.path.join(self.config['output_dir'], 'loss.png')
    fig.savefig(loss_fp)
    plt.close()
    
    # Learning Rate
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    ax.plot(learning_rates, marker = '.')
    ax.set_title("Learning Rate")
    ax.set_xlabel('Epoch')
    major_tick_spacing = max(1, len(learning_rates) // 10)
    ax.xaxis.set_major_locator(MultipleLocator(major_tick_spacing))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.set_ylabel('Learning Rate')
    ax.set_yscale('log')
    lr_fp = os.path.join(self.config['output_dir'], 'learning_rate.png')
    fig.savefig(lr_fp)
    plt.close()
    
  def train_epoch(self, dataloader, loss_fn, optimizer):
    self.model.train()
    
    train_loss = 0
    
    
    ## remove tqdm from here
    num_batches_todo = 1 + len(dataloader)
    num_batches_seen=0
    
    for X, y in dataloader:
      X, y = X.type('torch.FloatTensor').to(device), y.type('torch.FloatTensor').to(device)

      # Compute prediction error
      pred = self.model(X)
      loss = loss_fn(pred, y)
      train_loss += loss.item()
      num_batches_seen += 1

      # Backpropagation
      optimizer.zero_grad()
      loss.backward()

      optimizer.step()
    
    
#     with tqdm.tqdm(dataloader, unit = "batch", total = num_batches_todo) as tepoch:
#       for i, (X, y) in enumerate(tepoch):
#         X, y = X.type('torch.FloatTensor').to(device), y.type('torch.FloatTensor').to(device)
        
#         # Compute prediction error
#         pred = self.model(X)
#         loss = loss_fn(pred, y)
#         train_loss += loss.item()
#         num_batches_seen += 1

#         # Backpropagation
#         optimizer.zero_grad()
#         loss.backward()
        
#         optimizer.step()
#         loss_str = "%2.2f" % loss.item()
#         tepoch.set_postfix(loss=loss_str)
   
    train_loss = train_loss / num_batches_seen
#     print("Train loss: %f" % train_loss)
    return train_loss
  
  def predict(self, data, batch_size = 2048):
    ###
    self.model.eval()
    alldata= data
    
    predslist = []
    for i in range(0, np.shape(alldata)[0], batch_size):
      data = alldata[i:i+batch_size, :] # window to acommodate more hidden states without making edits to CUDA kernel
    
      with torch.no_grad():
        data = torch.from_numpy(data).type('torch.FloatTensor').to(device)
        preds = self.model(data)
        preds = preds.cpu().detach().numpy()
        predslist.append(preds)
        
    preds = np.concatenate(predslist)
    return preds
    
class MLP(nn.Module):
    def __init__(self, input_dims, output_dims, num_layers, hidden_size):
        super(MLP, self).__init__()
        assert num_layers > 1, "We assume we use a non-linear model, increase number of layers"
        
        layers = [nn.Linear(input_dims, hidden_size)]
        for i in range(num_layers-2):
          layers.append(nn.Linear(hidden_size, hidden_size))
          
        self.encoder = nn.ModuleList(layers)
        self.head = nn.Linear(hidden_size, output_dims)
        
    def forward(self, x):
        for layer in self.encoder:
          x = layer(x)
          x = nn.ReLU()(x)
        x = self.head(x)
        return x
      
      
      
      
class UMAP_DATASET(Dataset):
    def __init__(self, inputs, targets):
        self.data = inputs # [*, n_features]
        self.targets = targets # [*, 2] 
        
    def __len__(self):        
        return np.shape(self.data)[0]

    def __getitem__(self, index):
        data_item = self.data[index, :]
        targets_item = self.targets[index, :]
            
        return data_item, targets_item
