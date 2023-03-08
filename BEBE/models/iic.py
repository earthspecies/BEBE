import yaml
import numpy as np
import pickle
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.utils.data.dataset import Dataset
import torch.nn.functional as F
import torchmetrics
import tqdm
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
from BEBE.models.model_superclass import BehaviorModel
import scipy.special as special
import math
import pandas as pd
from pathlib import Path
from sys import float_info
import random

EPS = float_info.epsilon

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"

def _count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class iic(BehaviorModel):
  def __init__(self, config):
    super(iic, self).__init__(config)
    print(f"Using {device} device")
    
    self.downsizing_factor = self.model_config['downsizing_factor']
    self.lr = self.model_config['lr']
    self.weight_decay = self.model_config['weight_decay']
    self.n_epochs = self.model_config['n_epochs']
    self.context_window_samples = self.model_config['context_window_samples']
    self.batch_size = self.model_config['batch_size']
    self.dropout = self.model_config['dropout']
    self.blur_scale = self.model_config['blur_scale']
    self.jitter_scale = self.model_config['jitter_scale']
    self.n_clusters = self.config['num_clusters']
    self.temporal_window_samples = self.model_config['temporal_window_samples']
    self.conv_depth = self.model_config['conv_depth']
    self.ker_size = self.model_config['ker_size']
    self.dilation = self.model_config['dilation']
    self.hidden_size = self.model_config['hidden_size']
    self.n_heads = self.model_config['n_heads']
    
    torch.manual_seed(self.config['seed'])
    random.seed(self.config['seed'])
    np.random.seed(self.config['seed'])
    
    assert self.context_window_samples > 1, 'context window should be larger than 1'
    
    labels_bool = [x == 'label' for x in self.metadata['clip_column_names']]
    self.label_idx = [i for i, x in enumerate(labels_bool) if x][0] # int
    
    self.n_features = len(self.cols_included)
    self.encoder = Encoder(self.n_features,
                           self.conv_depth,
                           self.ker_size,
                           self.hidden_size,
                           self.dilation,
                           dropout = self.dropout,
                           blur_scale = self.blur_scale,
                           jitter_scale = self.jitter_scale).to(device)
    
    self.heads = nn.ModuleList([Head(self.hidden_size, self.n_clusters) for i in range(self.n_heads)]).to(device)
    
    print('Encoder parameters:')
    print(_count_parameters(self.encoder))
  
  def fit(self):
    dev_fps = self.config['dev_data_fp']
    
    dev_data = [self.load_model_inputs(fp) for fp in dev_fps]
    dev_dataset = BEHAVIOR_DATASET(dev_data, self.temporal_window_samples, True)
    dev_dataloader = DataLoader(dev_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers = 0)
    loss_fn = IICLoss(self.context_window_samples).to(device)
        
    optimizer = torch.optim.Adam([{'params' : self.encoder.parameters(), 'weight_decay' : self.weight_decay}, {'params' : self.heads.parameters(), 'weight_decay' : self.weight_decay}], lr=self.lr, amsgrad = True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.n_epochs, eta_min=self.lr / 100, last_epoch=- 1, verbose=False)
    
    dev_loss = []
    dev_loss_per_head = {i : [] for i in range(self.n_heads)}
    dev_predictions_loss = []
    learning_rates = []
    
    epochs = self.n_epochs
    for t in range(epochs):
        print(f"Epoch {t}\n-------------------------------")
        l, l_per_head = self.train_epoch(t, dev_dataloader, loss_fn, optimizer)
        dev_loss.append(l / self.n_heads)
        for i in range(self.n_heads):
          dev_loss_per_head[i].append(l_per_head[i])
       
        learning_rates.append(optimizer.param_groups[0]["lr"])
        scheduler.step()
    
    min_loss = min(l_per_head)
    chosen_head_index = l_per_head.index(min_loss)
    self.chosen_head = self.heads[i]
      
    print("Done!")
    
    ## Save training progress
    # Loss
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    
    ax.plot(dev_loss, label= 'average train loss', marker = '.')
    for i in dev_loss_per_head:
      ax.plot(dev_loss_per_head[i], label = 'loss for head '+str(i), marker = '.')
    
    ax.legend()
    ax.set_title("Cross Entropy Loss")
    ax.set_xlabel('Epoch')
    
    major_tick_spacing = max(1, len(dev_loss) // 10)
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
    
  def train_epoch(self, t, dataloader, loss_fn, optimizer):
    size = len(dataloader.dataset)
    self.encoder.train()
    self.heads.train()
    
    train_loss = 0
    num_batches_seen = 0
    train_loss_per_head = [0 for i in range(len(self.heads))]
    
    num_batches_todo = 1 + len(dataloader) // self.downsizing_factor
    with tqdm.tqdm(dataloader, unit = "batch", total = num_batches_todo) as tepoch:
      for i, X in enumerate(tepoch):
        if i == num_batches_todo :
          break
        num_batches_seen += 1
        X = X.to(device = device, dtype = torch.float)
        X = self.encoder(X)
        
        all_loss = None
        for i, head in enumerate(self.heads):
          probs = head(X)
          loss = loss_fn(probs) 
          if all_loss == None:
            all_loss = loss
          else:
            all_loss += loss
          train_loss += loss.item()
          train_loss_per_head[i] += loss.item()
       
        # Backpropagation
        optimizer.zero_grad()
        all_loss.backward()
        
        optimizer.step()
        loss_str = "%2.2f" % loss.item()
        tepoch.set_postfix(loss=loss_str)
    
    train_loss = train_loss / num_batches_seen
    train_loss_per_head = [l / num_batches_seen for l in train_loss_per_head]
    print("Train loss: %f" % train_loss)
    print("Train loss per head:")
    print(train_loss_per_head)
    return train_loss, train_loss_per_head
    
  def save(self):
    target_fp = os.path.join(self.config['final_model_dir'], "final_model.pickle")
    with open(target_fp, 'wb') as f:
      pickle.dump(self, f)

  def predict(self, data):
      self.encoder.eval()
      self.heads.eval()
      alldata= data

      predslist = []
      pred_len = self.temporal_window_samples
      
      for i in range(0, np.shape(alldata)[0], pred_len):
        data = alldata[i:i+pred_len, :] 

        with torch.no_grad():
          data = np.expand_dims(data, axis =0)
          data = torch.from_numpy(data).type('torch.FloatTensor').to(device)
          x = self.encoder(data)
          preds = self.chosen_head(x)
          preds = preds.cpu().detach().numpy()
          preds = np.squeeze(preds, axis = 0)
          preds = np.argmax(preds, axis = -1).astype(np.uint8)
          predslist.append(preds)

      preds = np.concatenate(predslist)
      
      return preds, None  

class BEHAVIOR_DATASET(Dataset):
    def __init__(self, data, temporal_window_samples, train):
        self.temporal_window = temporal_window_samples
        self.data = data
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
        self.train = train
        
    def __len__(self):        
        return self.data_points - len(self.data) * self.temporal_window

    def __getitem__(self, index):
        clip_number = np.where(index >= self.data_start_indices)[0][-1] #which clip do I draw from?
        data_item = self.data[clip_number]
        start = index - self.data_start_indices[clip_number]
        end = start+ self.temporal_window
        data_item = data_item[start:end, :]   
        return torch.from_numpy(data_item)

class IICLoss(nn.Module):
  def __init__(self, context_window_samples):
    super(IICLoss, self).__init__()
    self.half_T_side_dense = context_window_samples // 2
    
  def forward(self, probs):
    probs = torch.transpose(probs, -1, -2) #[batch, classes, len]
    loss, _ = IID_segmentation_loss(probs, probs, lamb = 1.0, half_T_side_dense = self.half_T_side_dense)
    return loss

# modified from:
# https://github.com/xu-ji/IIC
def IID_segmentation_loss(x1_outs, x2_outs, lamb=1.0,
                          half_T_side_dense=None):
  assert (x1_outs.requires_grad)
  assert (x2_outs.requires_grad)

  # zero out all irrelevant patches
  bn, k, seq_len = x1_outs.shape

  # sum over everything except classes, by convolving x1_outs with x2_outs_inv
  # which is symmetric, so doesn't matter which one is the filter
  x1_outs = x1_outs.permute(1, 0, 2).contiguous()  # k, ni, seq_len
  x2_outs = x2_outs.permute(1, 0, 2).contiguous()  # k, ni, seq_len

  # k, k, 2 * half_T_side_dense + 1
  p_i_j = F.conv1d(x1_outs, weight=x2_outs, padding= half_T_side_dense)
  p_i_j = p_i_j.sum(dim=2, keepdim=False)  # k, k

  # normalise, use sum, not bn * h * w * T_side * T_side, because we use a mask
  # also, some pixels did not have a completely unmasked box neighbourhood,
  # but it's fine - just less samples from that pixel
  current_norm = float(p_i_j.sum())
  p_i_j = p_i_j / current_norm

  # symmetrise
  p_i_j = (p_i_j + p_i_j.t()) / 2.

  # compute marginals
  p_i_mat = p_i_j.sum(dim=1).unsqueeze(1)  # k, 1
  p_j_mat = p_i_j.sum(dim=0).unsqueeze(0)  # 1, k

  # for log stability; tiny values cancelled out by mult with p_i_j anyway
  p_i_j[(p_i_j < EPS).data] = EPS
  p_i_mat[(p_i_mat < EPS).data] = EPS
  p_j_mat[(p_j_mat < EPS).data] = EPS

  # maximise information
  loss = (-p_i_j * (torch.log(p_i_j) - lamb * torch.log(p_i_mat) -
                    lamb * torch.log(p_j_mat))).sum()

  # for analysis only
  loss_no_lamb = (-p_i_j * (torch.log(p_i_j) - torch.log(p_i_mat) -
                            torch.log(p_j_mat))).sum()

  return loss, loss_no_lamb
  
class Encoder(nn.Module):
    def __init__(self, n_features, conv_depth, ker_size, hidden_size, dilation, dropout, blur_scale = 0, jitter_scale = 0):
        super(Encoder, self).__init__()
        self.blur_scale = blur_scale
        self.jitter_scale = jitter_scale
        
        self.bn = nn.BatchNorm1d(n_features)
        self.conv = [_conv_block_1d(n_features, hidden_size, ker_size, dilation = dilation)]
        for i in range(conv_depth - 1):
          self.conv.append(_conv_block_1d(hidden_size, hidden_size, ker_size, dilation = dilation))
        
        self.conv = nn.ModuleList(self.conv)
        
    def forward(self, x):
        # X is [batch, seq_len, channels]
        seq_len = x.size()[-2]
        
        x = torch.transpose(x, -1, -2)
        x = self.bn(x)
        
        if self.training:
          # Perform augmentations to normalized data
          size = x.size()
          if self.blur_scale:
            blur = self.blur_scale * torch.randn(size, device = x.device)
          else:
            blur = 0.
          if self.jitter_scale:
            jitter = self.jitter_scale *torch.randn((size[0], 1, size[2]), device = x.device)
          else:
            jitter = 0.
          x = x + blur + jitter 
        
        for block in self.conv:
          x = block(x)
          
        return x
      
class Head(nn.Module):
    def __init__(self, hidden_size, n_classes, softmax = True):
        super(Head, self).__init__()
        self.linear = nn.Linear(hidden_size, n_classes)
        self.out = torch.nn.Softmax(dim = -1) if softmax else torch.nn.Identity()
        
    def forward(self, x):
        x = torch.transpose(x, -1, -2)
        x = self.linear(x)
        x = self.out(x)
        return x

def _conv_block_1d(in_channels, out_channels, kernel_size, dilation = 1):
  block = nn.Sequential(
    nn.Conv1d(in_channels, out_channels, kernel_size, dilation = dilation, padding='same', bias=False),
    torch.nn.BatchNorm1d(out_channels),
    nn.ReLU()
  )
  return block
