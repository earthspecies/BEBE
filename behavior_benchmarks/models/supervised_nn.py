import yaml
import numpy as np
import pickle
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchmetrics
from behavior_benchmarks.models.supervised_nn_utils import BEHAVIOR_DATASET
import tqdm
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
from behavior_benchmarks.models.model_superclass import BehaviorModel
from behavior_benchmarks.applications.S4.S4 import S4

import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"

def _count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class supervised_nn(BehaviorModel):
  def __init__(self, config):
    super(supervised_nn, self).__init__(config)
    print(f"Using {device} device")
    # self.config = config
    # self.read_latents = config['read_latents']
    # self.model_config = config['supervised_nn_config']
    # self.metadata = config['metadata']
    self.unknown_label = config['metadata']['label_names'].index('unknown')
    
    ##
    self.downsizing_factor = self.model_config['downsizing_factor']
    self.lr = self.model_config['lr']
    self.weight_decay = self.model_config['weight_decay']
    self.n_epochs = self.model_config['n_epochs']
    self.hidden_size = self.model_config['hidden_size']
    self.n_s4_blocks = self.model_config['num_layers']
    self.temporal_window_samples = self.model_config['temporal_window_samples']
    self.batch_size = self.model_config['batch_size']
    self.dropout = self.model_config['dropout']
    self.blur_scale = self.model_config['blur_scale']
    self.jitter_scale = self.model_config['jitter_scale']
    self.rescale_param = self.model_config['rescale_param']
    self.sparse_annotations = self.model_config['sparse_annotations']
    self.weight_factor = self.model_config['weight_factor']
    self.state_size = self.model_config['state_size']
    ##
    
    # cols_included_bool = [x in self.config['input_vars'] for x in self.metadata['clip_column_names']] 
    # self.cols_included = [i for i, x in enumerate(cols_included_bool) if x]
    
    labels_bool = [x == 'label' for x in self.metadata['clip_column_names']]
    self.label_idx = [i for i, x in enumerate(labels_bool) if x][0] # int
    
    self.n_classes = len(self.metadata['label_names']) 
    self.n_features = len(self.cols_included)
    
    self.model = Classifier(self.n_features,
                            self.n_classes,
                            hidden_size = self.hidden_size,
                            state_size = self.state_size,
                            n_s4_blocks = self.n_s4_blocks,
                            dropout = self.dropout,
                            blur_scale = self.blur_scale,
                            jitter_scale = self.jitter_scale).to(device)
    
    #print(self.model)
    print('Model parameters:')
    print(_count_parameters(self.model))
  
  def load_model_inputs(self, filepath, read_latents = False):
    if read_latents:
      raise NotImplementedError("Supervised model is expected to read from raw data")
      #return np.load(filepath)
    else:
      return np.load(filepath)[:, self.cols_included] #[n_samples, n_features]
    
  def load_labels(self, filepath):
    labels = np.load(filepath)[:, self.label_idx].astype(int)
    return labels 
    
  def fit(self):
    ## get data. assume stored in memory for now
    if self.read_latents:
      train_fps = self.config['train_data_latents_fp']
      val_fps = self.config['val_data_latents_fp']
      test_fps = self.config['test_data_latents_fp']
    else:
      train_fps = self.config['train_data_fp']
      val_fps = self.config['val_data_fp']
      test_fps = self.config['test_data_fp']
    
    train_data = [self.load_model_inputs(fp, read_latents = self.read_latents) for fp in train_fps]
    val_data = [self.load_model_inputs(fp, read_latents = self.read_latents) for fp in val_fps]
    test_data = [self.load_model_inputs(fp, read_latents = self.read_latents) for fp in test_fps]
    
    train_labels = [self.load_labels(fp) for fp in train_fps]
    val_labels = [self.load_labels(fp) for fp in val_fps]
    test_labels = [self.load_labels(fp) for fp in test_fps]
    
    train_dataset = BEHAVIOR_DATASET(train_data, train_labels, True, self.temporal_window_samples, self.config, rescale_param = self.rescale_param)
    if self.sparse_annotations:
      indices_to_keep = train_dataset.get_annotated_windows()
      train_dataset = Subset(train_dataset, indices_to_keep)  
      print("Number windowed train examples after subselecting: %d" % len(train_dataset))
    train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers = 0)
    
    val_dataset = BEHAVIOR_DATASET(val_data, val_labels, False, self.temporal_window_samples, self.config)
    if self.sparse_annotations:
      indices_to_keep = val_dataset.get_annotated_windows()
      val_dataset = Subset(val_dataset, indices_to_keep) 
      
    num_examples_val = len(list(range(0, len(val_dataset), self.downsizing_factor)))
    val_dataset = Subset(val_dataset, list(range(0, len(val_dataset), self.downsizing_factor)))
    print("Number windowed val examples after subselecting: %d" % len(val_dataset))
    val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False, num_workers = 0)
    
    test_dataset = BEHAVIOR_DATASET(test_data, test_labels, False, self.temporal_window_samples, self.config)
    if self.sparse_annotations:
      indices_to_keep = test_dataset.get_annotated_windows()
      test_dataset = Subset(test_dataset, indices_to_keep)

    num_examples_test = len(list(range(0, len(test_dataset), self.downsizing_factor)))
    test_dataset = Subset(test_dataset, list(range(0, len(test_dataset), self.downsizing_factor)))
    print("Number windowed test examples after subselecting: %d" % len(test_dataset))
    test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False, num_workers = 0)
    
    proportions = train_dataset.get_class_proportions()
    weight = (proportions ** self.weight_factor).to(device)
    loss_fn = nn.CrossEntropyLoss(ignore_index = self.unknown_label, weight = weight)
    
    loss_fn_no_reduce = nn.CrossEntropyLoss(ignore_index = self.unknown_label, reduction = 'sum', weight = weight)
    optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay, amsgrad = True)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.n_epochs, eta_min=0, last_epoch=- 1, verbose=False)
    
    train_loss = []
    test_loss = []
    val_loss = []
    train_acc = []
    test_acc = []
    val_acc = []
    learning_rates = []
    
    epochs = self.n_epochs
    for t in range(epochs):
        print(f"Epoch {t}\n-------------------------------")
        l, a = self.train_epoch(train_dataloader, loss_fn, optimizer)
        train_loss.append(l)
        train_acc.append(a)
        l, a = self.test_epoch(val_dataloader, loss_fn_no_reduce, name = "Val", loss_denom = num_examples_val * self.temporal_window_samples)
        val_loss.append(l)
        val_acc.append(a)
        l, a = self.test_epoch(test_dataloader, loss_fn_no_reduce, name = "Test", loss_denom = num_examples_test* self.temporal_window_samples)
        test_loss.append(l)
        test_acc.append(a)
        
        learning_rates.append(optimizer.param_groups[0]["lr"])
        scheduler.step()
      
    print("Done!")
    
    ## Save training progress
    
    # Loss
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    
    ax.plot(train_loss, label= 'train', marker = '.')
    ax.plot(val_loss, label= 'val', marker = '.')
    ax.plot(test_loss, label = 'test', marker = '.')
    ax.legend()
    ax.set_title("Cross Entropy Loss")
    ax.set_xlabel('Epoch')
    
    major_tick_spacing = max(1, len(train_loss) // 10)
    ax.xaxis.set_major_locator(MultipleLocator(major_tick_spacing))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.set_ylabel('Loss')
    loss_fp = os.path.join(self.config['output_dir'], 'loss.png')
    fig.savefig(loss_fp)
    plt.close()

    # Accuracy
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    ax.plot(train_acc, label= 'train', marker = '.')
    ax.plot(val_acc, label= 'val', marker = '.')
    ax.plot(test_acc, label = 'test', marker = '.')
    ax.legend()
    ax.set_title("Mean accuracy")
    ax.set_xlabel('Epoch')
    major_tick_spacing = max(1, len(train_acc) // 10)
    ax.xaxis.set_major_locator(MultipleLocator(major_tick_spacing))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.set_ylabel('Accuracy')
    acc_fp = os.path.join(self.config['output_dir'], 'acc.png')
    fig.savefig(acc_fp)
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
    size = len(dataloader.dataset)
    self.model.train()
    # Use accuracy instead of f1, since torchmetrics doesn't handle masking for precision as we want it to
    acc_score = torchmetrics.Accuracy(num_classes = self.n_classes, average = 'macro', mdmc_average = 'global', ignore_index = self.unknown_label)
    train_loss = 0
    num_batches_seen = 0
    
    num_batches_todo = 1 + len(dataloader) // self.downsizing_factor
    with tqdm.tqdm(dataloader, unit = "batch", total = num_batches_todo) as tepoch:
      for i, (X, y) in enumerate(tepoch):
        if i == num_batches_todo :
          break
        num_batches_seen += 1
        X, y = X.type('torch.FloatTensor').to(device), y.type('torch.LongTensor').to(device)
        
        # Compute prediction error
        pred = self.model(X)
        loss = loss_fn(pred, y)
        train_loss += loss.item()

        acc_score.update(pred.cpu(), y.cpu())

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        
        # torch.nn.utils.clip_grad_norm(self.model.parameters(), 0.45)
        
        # total_norm = 0
        # for p in self.model.parameters():
        #     param_norm = p.grad.detach().data.norm(2)
        #     total_norm += param_norm.item() ** 2
        # total_norm = total_norm ** 0.5
        # print(total_norm)
        
        optimizer.step()
        loss_str = "%2.2f" % loss.item()
        tepoch.set_postfix(loss=loss_str)
        
    acc = acc_score.compute()
    train_loss = train_loss / num_batches_seen
    print("Train loss: %f, Train accuracy: %f" % (train_loss, acc))
    return train_loss, acc
    
  def test_epoch(self, dataloader, loss_fn, name = "Test", loss_denom = 0):
    size = len(dataloader.dataset)
    self.model.eval()
    test_loss = 0
    # Use accuracy instead of f1, since torchmetrics doesn't handle masking for precision as we want it to
    acc_score = torchmetrics.Accuracy(num_classes = self.n_classes, average = 'macro', mdmc_average = 'global', ignore_index = self.unknown_label)
    
    with torch.no_grad():
        num_batches_todo = 1 + len(dataloader) // self.downsizing_factor
        for i, (X, y) in enumerate(dataloader):
            X, y = X.type('torch.FloatTensor').to(device), y.type('torch.LongTensor').to(device)
            pred = self.model(X)
            acc_score.update(pred.cpu(), y.cpu())
            test_loss += loss_fn(pred, y).item()
    test_loss /= loss_denom
    acc = acc_score.compute()
    print("%s loss: %f, %s accuracy: %f" % (name, test_loss, name, acc))
    return test_loss, acc
    
  def save(self):
    target_fp = os.path.join(self.config['final_model_dir'], "final_model.pickle")
    with open(target_fp, 'wb') as f:
      pickle.dump(self, f)
  
  def predict(self, data):
    ###
    self.model.eval()
    alldata= data
    
    predslist = []
    pred_len = 10000
    for i in range(0, np.shape(alldata)[0], pred_len):
      data = alldata[i:i+pred_len, :] # window to acommodate more hidden states without making edits to CUDA kernel
    
      with torch.no_grad():
        data = np.expand_dims(data, axis =0)
        data = torch.from_numpy(data).type('torch.FloatTensor').to(device)
        preds = self.model(data)
        preds = preds.cpu().detach().numpy()
        preds = np.squeeze(preds, axis = 0)
        preds = np.argmax(preds, axis = 0).astype(np.uint8)
        
        predslist.append(preds)
    preds = np.concatenate(predslist)
    return preds, None  
      
class S4Block(nn.Module):
    def __init__(self, H, N, dropout= 0.):
      super(S4Block, self).__init__()
      self.bn1 = nn.BatchNorm1d(H)
      self.s4 = S4(H, d_state = N, bidirectional = True, dropout = dropout)
      self.linear1 = nn.Conv1d(H, H, 1)
      self.bn2 = nn.BatchNorm1d(H)
      self.linear2 = nn.Conv1d(H, 2*H, 1)
      self.linear3 = nn.Conv1d(2*H, H, 1)
      
    def forward(self, x):
      y = x
      x = self.bn1(x)
      x = self.s4(x)[0]
      x = y+ x
      
      y = x
      x = self.bn2(x)
      x = self.linear2(x)
      x = nn.functional.gelu(x)
      x = self.linear3(x)
      x = y+ x
      return x

#v6 lol
class Classifier(nn.Module):
    def __init__(self, n_features, n_classes, hidden_size, state_size, n_s4_blocks, dropout, blur_scale = 0, jitter_scale = 0):
        super(Classifier, self).__init__()
        self.blur_scale = blur_scale
        self.jitter_scale = jitter_scale
        
        self.embedding = nn.Conv1d(n_features, hidden_size, 1, padding = 'same', bias = False) 
        self.head = nn.Conv1d(hidden_size, n_classes, 1, padding = 'same')
        #self.gru = nn.GRU(conv_features, hidden_size, num_layers = num_layers_gru, bidirectional = True, batch_first = True, dropout = dropout)
          
        self.s4_blocks = [S4Block(hidden_size, state_size, dropout = dropout) for i in range(n_s4_blocks)]
        self.s4_blocks = nn.ModuleList(self.s4_blocks)
        
    def forward(self, x):
        
        x = torch.transpose(x, 1,2) # [batch, seq_len, n_features] -> [batch, n_features, seq_len]
        if self.training:
          # Perform augmentations to normalized data
          size = x.size()
          if self.blur_scale:
            blur = self.blur_scale * torch.randn(size, device = x.device)
          else:
            blur = 0.
          if self.jitter_scale:
            jitter = self.jitter_scale *torch.randn((size[0], size[1], 1), device = x.device)
          else:
            jitter = 0.
          x = x + blur + jitter 
        
        x = self.embedding(x)
        for block in self.s4_blocks:
          x = block(x)
        
        logits = self.head(x)
        return logits  

# # # v4:  
# class LSTM_Classifier(nn.Module):
#     def __init__(self,
#                  n_features,
#                  n_classes,
#                  hidden_size,
#                  num_layers_lstm,
#                  dropout,
#                  blur_scale = 0,
#                  jitter_scale = 0,
#                  n_fft = 128,
#                  hop_length = 32,
#                  power = 1.0,
#                  spec_ker_size = 3,
#                  ts_ker_size = 5):
#         super(LSTM_Classifier, self).__init__()
#         self.blur_scale = blur_scale
#         self.jitter_scale = jitter_scale
        
#         #
#         self.bn = torch.nn.BatchNorm1d(n_features)
        
#         # Spectrogram 
#         self.hop_length = hop_length
#         self.n_fft = n_fft
#         n_freq_bins = n_fft //2 + 1

#         self.spectrogram = T.Spectrogram(
#             n_fft=self.n_fft,
#             win_length=None,
#             hop_length=self.hop_length,
#             center=True,
#             pad_mode="reflect",
#             power=power,
#         )
        
#         # Spectrogram fe params
#         n_channels_out_spec = 64
#         ker_size = spec_ker_size
#         self.spec_fe = Spectrogram_FeatureExtractor(n_freq_bins, n_channels_out_spec, ker_size)
        
#         # Time sereies fe
#         ker_width = ts_ker_size
#         n_channels_out_ts = 128
#         self.ts_fe = TimeSeries_FeatureExtractor(n_features, n_channels_out_ts, ker_width)
        
#         into_lstm_channels = n_features*n_channels_out_spec + n_channels_out_ts
#         self.lstm = nn.LSTM(into_lstm_channels, hidden_size, num_layers = num_layers_lstm, bidirectional = True, batch_first = True, dropout = dropout)
#         self.head = nn.Conv1d(2* hidden_size, n_classes, 1, padding = 'same')
        

#     def forward(self, x):
#         # x: [batch, seq_len, num_feat]
        
#         seq_len = x.size()[1]
#         num_feat = x.size()[2]
#         n_time_bins_spec = seq_len // self.hop_length + 1
        
#         ## setup
#         x = torch.transpose(x, 1,2)
#         x = self.bn(x)
        
#         if self.training:
#           # Perform augmentations to normalized data
#           blur = self.blur_scale * torch.randn(x.size(), device = x.device)
#           x = x + blur
        
        
#         ## Generate spectral representations
#         # x_spec is a list of tensors of shape (batch, n_freq_bins, n_time_bins)
#         # where n_freq_bins = n_fft // 2 + 1
#         # and n_time_bins = seq_len // hop_len + 1
#         x_spec = [self.spectrogram(x[:, i, :]) for i in range(num_feat)]
        
#         x_spec = [torch.unsqueeze(y, 2) for y in x_spec]
#         x_spec = [self.dropout(y) for y in x_spec]
#         x_spec = [torch.transpose(y, 1,2) for y in x_spec]
        
        
        
#         #x_spec = [torch.unsqueeze(y, 1) for y in x_spec]
        
#         ## Extract spectrogram features
#         x_spec = [self.spec_fe(y) for y in x_spec] # list of tensors of shape (batch, n_channels, n_time_bins)
        
#         ## Extract time series features
#         x_ts = self.ts_fe(x)
        
#         # resample to duration of x_spec
#         x_ts = nn.functional.interpolate(x_ts, size=n_time_bins_spec, mode='nearest-exact')
        
#         # concat
#         x = torch.cat(x_spec + [x_ts], axis = 1)
        
#         # lstm
#         x = torch.transpose(x, 1,2) # [batch, n_features, seq_len] -> [batch, seq_len, n_features]
#         x = self.lstm(x)[0]
#         x = torch.transpose(x, 1,2) # [batch, seq_len, n_features] -> [batch, n_features, seq_len]
        
#         # make final predictions and upsample
#         logits = self.head(x)
#         logits = nn.functional.interpolate(logits, size=seq_len, mode='nearest-exact') 
                            
#         return logits

# class TimeSeries_FeatureExtractor(nn.Module):
#     def __init__(self, n_channels_in, n_channels_out, ker_width):
#         super(TimeSeries_FeatureExtractor, self).__init__()
#         layer1_in_channels = n_channels_in
#         layer1_out_channels = 128
#         layer2_in_channels = layer1_in_channels + layer1_out_channels
#         layer2_out_channels = 128
#         layer3_in_channels = layer2_in_channels + layer2_out_channels
#         layer3_out_channels = 128
#         layer4_in_channels = layer3_in_channels + layer3_out_channels
#         layer4_out_channels = 128
        
#         head_in_channels = layer4_in_channels + layer4_out_channels
#         head_out_channels = n_channels_out
        
#         self.layer1 = _conv_block_1d(layer1_in_channels, layer1_out_channels, ker_width)
#         self.layer2 = _conv_block_1d(layer1_out_channels, layer2_out_channels, ker_width)
#         self.layer3 = _conv_block_1d(layer2_out_channels, layer3_out_channels, ker_width)
#         self.layer4 = _conv_block_1d(layer3_out_channels, layer4_out_channels, ker_width)
#         self.head = _conv_block_1d(layer4_out_channels, head_out_channels, ker_width)
        

#     def forward(self, x):
#         # x: tensor of shape [batch, n_channels_in, seq_len]
#         # returns time_series_features: tensor of shape [batch, n_channels_out, out_seq_len]
#         # where out_seq_len is approximately seq_len // 16
        
#         x1 = self.layer1(x) # conv, bn, relu -> (batch, layer1_out_channels, seq_len)
#         x = x1 #torch.cat([x,x1], axis = 1) # concat -> (batch, layer2_in_channels, seq_len)
#         x = nn.AvgPool1d(2, padding=1)(x)# average_pool -> (batch, layer2_in_channels, seq_len //2 +1)
#         x2 = self.layer2(x) # conv, bn, relu -> (batch, layer2_out_channels, ...)
#         x = x2 + x #torch.cat([x,x2], axis = 1) # concat -> (batch, layer3_in_channels, ...)
#         x = nn.AvgPool1d(2, padding=1)(x)# average_pool -> (batch, layer3_in_channels, ...)
#         x3 = self.layer3(x) # conv, bn, relu -> (batch, layer3_out_channels, ...)
#         x = x3 + x #torch.cat([x, x3], axis =1)# concat -> (batch, layer4_in_channels, ...)
#         x = nn.AvgPool1d(2, padding=1)(x)# average_pool -> (batch, layer4_in_channels, ...)
#         x4 = self.layer4(x)# conv, bn, relu -> (batch, layer4_out_channels, ...)
#         x = x4 + x #torch.cat([x,x4], axis = 1)# concat -> (batch, head_in_channels, ...)
#         x = nn.AvgPool1d(2, padding=1)(x)# average_pool -> (batch, head_in_channels, ...)
#         time_series_features = self.head(x) # conv, bn, relu -> (batch, head_out_channels, ...)
#         return time_series_features
      
# class Spectrogram_FeatureExtractor(nn.Module):
#     def __init__(self, n_freq_bins, n_channels_out, ker_size):
#         super(Spectrogram_FeatureExtractor, self).__init__()
        
#         layer1_in_channels = 1
#         layer1_out_channels = 64
#         layer2_in_channels = layer1_in_channels + layer1_out_channels
#         layer2_out_channels = 64
#         layer3_in_channels = layer2_in_channels + layer2_out_channels
#         layer3_out_channels = 64
#         layer4_in_channels = layer3_in_channels + layer3_out_channels
#         layer4_out_channels = 64
        
#         head_in_channels = layer4_in_channels + layer4_out_channels
#         head_out_channels = n_channels_out
        
#         self.layer1 = _conv_block_2d(layer1_in_channels, layer1_out_channels, ker_size)
#         self.layer2 = _conv_block_2d(layer1_out_channels, layer2_out_channels, ker_size)
#         self.layer3 = _conv_block_2d(layer2_out_channels, layer3_out_channels, ker_size)
#         self.layer4 = _conv_block_2d(layer3_out_channels, layer4_out_channels, ker_size)
#         self.head = _conv_block_2d(layer4_out_channels, head_out_channels, ker_size)
#         self.bn = torch.nn.BatchNorm2d(1)
        

#     def forward(self, x):
#         # x: tensor of shape (batch, 1, n_freq_bins, n_time_bins)
#         # returns spectrogram_features: tensor of shape (batch, n_channels_out, n_time_bins)
        
#         x = self.bn(x)
#         x1 = self.layer1(x) # 3x3 conv, bn, relu -> (batch, layer1_out_channels, n_freq_bins, n_time_bins)
#         x = x1 #torch.cat([x1, x], axis = 1)# concat -> (batch, layer2_in_channels, n_freq_bins, n_time_bins)
#         x = nn.AvgPool2d((2,1), padding=(1, 0))(x)# mean_pool -> (batch, layer2_in_channels, n_freq_bins //2 + 1, n_time_bins)
#         x2 = self.layer2(x) # 3x3 conv, bn, relu -> (batch, layer2_out_channels, n_freq_bins //2 + 1, n_time_bins)
#         x = x +x2 #torch.cat([x2, x], axis = 1) # concat -> (batch, layer3_in_channels, n_freq_bins //2 + 1, n_time_bins)
#         x = nn.AvgPool2d((2,1), padding=(1, 0))(x) # mean_pool -> (batch, layer3_in_channels, .., n_time_bins)
#         x3 = self.layer3(x) # 3x3 conv, bn, relu -> (batch, layer3_out_channels, ..., n_time_bins)
#         x = x +x3 #torch.cat([x3, x], axis = 1)# concat -> (batch, layer4_in_channels, ..., n_time_bins)
#         x = nn.AvgPool2d((2,1), padding=(1, 0))(x) # mean_pool -> (batch, layer4_in_channels, ..., n_time_bins)
#         x4 = self.layer4(x) # 3x3 conv, bn, relu -> (batch, layer4_out_channels, ..., n_time_bins
#         x = x + x4 #torch.cat([x4, x], axis = 1) # concat -> (batch, head_in_channels, ..., n_time_bins)
#         x = self.head(x) # 1x1 conv, bn, relu -> (batch, n_channels_out, ..., n_time_bins)
#         x_height = x.size()[2]
#         x = nn.AvgPool2d((x_height,1), padding=0)(x) # mean_pool -> (batch, n_channels_out, 1, n_time_bins)
        
#         spectrogram_features = torch.squeeze(x, 2) # -> (batch, n_channels_out, n_time_bins)
#         return spectrogram_features

# def _conv_block_2d(in_channels, out_channels, kernel_size):
#   block = nn.Sequential(
#     nn.Conv2d(in_channels, out_channels, kernel_size, padding='same', bias=False),
#     torch.nn.BatchNorm2d(out_channels),
#     nn.ReLU(),
#     nn.Conv2d(out_channels, out_channels, kernel_size, padding='same', bias=False),
#     torch.nn.BatchNorm2d(out_channels),
#     nn.ReLU()
#   )
#   return block

# def _conv_block_1d(in_channels, out_channels, kernel_width):
#   block = nn.Sequential(
#     nn.Conv1d(in_channels, out_channels, kernel_width, padding='same', bias=False),
#     torch.nn.BatchNorm1d(out_channels),
#     nn.ReLU(),
#     nn.Conv1d(out_channels, out_channels, kernel_width, padding='same', bias=False),
#     torch.nn.BatchNorm1d(out_channels),
#     nn.ReLU()
#   )
#   return block