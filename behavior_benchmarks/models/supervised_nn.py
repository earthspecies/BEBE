import yaml
import numpy as np
import pickle
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchmetrics
from behavior_benchmarks.models.pytorch_dataloaders import BEHAVIOR_DATASET
import tqdm
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

def _count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class supervised_nn():
  def __init__(self, config):
    self.config = config
    self.read_latents = config['read_latents']
    self.model_config = config['supervised_nn_config']
    self.metadata = config['metadata']
    self.unknown_label = config['metadata']['label_names'].index('unknown')
    
    ##
    self.downsizing_factor = self.model_config['downsizing_factor']
    self.lr = self.model_config['lr']
    self.weight_decay = self.model_config['weight_decay']
    self.scheduler_epochs_between_step = self.model_config['scheduler_epochs_between_step']
    self.n_epochs = self.model_config['n_epochs']
    self.hidden_size = self.model_config['hidden_size']
    self.num_layers = self.model_config['num_layers']
    self.temporal_window_samples = self.model_config['temporal_window_samples']
    self.batch_size = self.model_config['batch_size']
    self.dropout = self.model_config['dropout']
    ##
    
    cols_included_bool = [x in self.config['input_vars'] for x in self.metadata['clip_column_names']] 
    self.cols_included = [i for i, x in enumerate(cols_included_bool) if x]
    
    labels_bool = [x == 'label' for x in self.metadata['clip_column_names']]
    self.label_idx = [i for i, x in enumerate(labels_bool) if x][0] # int
    
    self.n_classes = len(self.metadata['label_names']) 
    self.n_features = len(self.cols_included)
    
    self.model = LSTM_Classifier(self.n_features, self.n_classes, self.hidden_size, self.num_layers, self.dropout).to(device)
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
    
    train_dataset = BEHAVIOR_DATASET(train_data, train_labels, True, self.temporal_window_samples)
    train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers = 0)
    
    val_dataset = BEHAVIOR_DATASET(val_data, val_labels, False, self.temporal_window_samples)
    num_examples_val = len(list(range(0, len(val_dataset), self.downsizing_factor)))
    val_dataset = Subset(val_dataset, list(range(0, len(val_dataset), self.downsizing_factor)))
    val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False, num_workers = 0)
    
    test_dataset = BEHAVIOR_DATASET(test_data, test_labels, False, self.temporal_window_samples)
    num_examples_test = len(list(range(0, len(test_dataset), self.downsizing_factor)))
    test_dataset = Subset(test_dataset, list(range(0, len(test_dataset), self.downsizing_factor)))
    test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False, num_workers = 0)
    
    loss_fn = nn.CrossEntropyLoss(ignore_index = self.unknown_label)
    loss_fn_no_reduce = nn.CrossEntropyLoss(ignore_index = self.unknown_label, reduction = 'sum')
    optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay, amsgrad = True)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, self.scheduler_epochs_between_step, gamma=0.1, last_epoch=- 1, verbose=False)
    
    train_loss = []
    test_loss = []
    val_loss = []
    train_acc = []
    test_acc = []
    val_acc = []
    
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
        scheduler.step()
    print("Done!")
    
    # Save training progress
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    
    ax.plot(train_loss, label= 'train', marker = '.')
    ax.plot(val_loss, label= 'train', marker = '.')
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
    
    # Save training progress
    
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    ax.plot(train_acc, label= 'train', marker = '.')
    ax.plot(val_acc, label= 'train', marker = '.')
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
    
    ###
    
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
    with torch.no_grad():
      data = data
      data = np.expand_dims(data, axis =0)
      data = torch.from_numpy(data).type('torch.FloatTensor').to(device)
      preds = self.model(data)
      preds = preds.cpu()
      preds = preds.squeeze(axis = 0)
      preds = np.argmax(preds, axis = 0)
    return preds, None
    ###
  
  def predict_from_file(self, fp):
    inputs = self.load_model_inputs(fp, read_latents = self.read_latents)
    predictions, latents = self.predict(inputs)
    return predictions, latents

class LSTM_Classifier(nn.Module):
    def __init__(self, n_features, n_classes, hidden_size, num_layers, dropout):
        super(LSTM_Classifier, self).__init__()
        
        self.head = nn.Conv1d(2*hidden_size, n_classes, 1, padding = 'same')
        self.lstm = nn.LSTM(n_features + 64, hidden_size, num_layers = num_layers, bidirectional = True, batch_first = True, dropout = dropout)
        
        self.conv1 = nn.Sequential(
          nn.Conv1d(n_features,64,7, padding = 'same'),
          torch.nn.BatchNorm1d(64),
          nn.ReLU(),
          nn.Conv1d(64,64,7, padding = 'same'),
          torch.nn.BatchNorm1d(64),
          nn.ReLU()
        )
        
        self.conv2 = nn.Sequential(
          nn.Conv1d(n_features + 64,64,7, padding = 'same'),
          torch.nn.BatchNorm1d(64),
          nn.ReLU(),
          nn.Conv1d(64,64,7, padding = 'same'),
          torch.nn.BatchNorm1d(64),
          nn.ReLU()
        )
        
        self.bn = torch.nn.BatchNorm1d(n_features)
        self.dropout = nn.Dropout(p = 0.01)
    
#         self.down1 = nn.Sequential( #512->256
#           nn.Conv1d(n_features,16,7, padding = 'same'),
#           torch.nn.BatchNorm1d(16),
#           nn.ReLU(),
#           nn.MaxPool1d(2, stride=2)
#         )
      
#         self.down2 = nn.Sequential( #256-> 128
#           nn.Conv1d(16,32,3, padding = 'same'),
#           torch.nn.BatchNorm1d(32),
#           nn.ReLU(),
#           nn.MaxPool1d(2, stride=2)
#         )
        
#         self.down3 = nn.Sequential( #128->64
#           nn.Conv1d(32,64,3, padding = 'same'),
#           torch.nn.BatchNorm1d(64),
#           nn.ReLU(),
#           nn.MaxPool1d(2, stride=2)
#         )
        
#         self.down4 = nn.Sequential( #64->32
#           nn.Conv1d(64,128,3, padding = 'same'),
#           torch.nn.BatchNorm1d(128),
#           nn.ReLU(),
#           nn.MaxPool1d(2, stride=2)
#         )
        
#         self.up4 =  nn.Sequential( #32->64
#           nn.ConvTranspose1d(128,64,2, stride = 2),
#           torch.nn.BatchNorm1d(64),
#           nn.ReLU()
#         )
        
#         self.up3 =  nn.Sequential( #64->128
#           nn.ConvTranspose1d(128,128,2, stride = 2),
#           torch.nn.BatchNorm1d(128),
#           nn.ReLU()
#         )
        
#         self.up2 =  nn.Sequential( #128->256
#           nn.ConvTranspose1d(160,128,2, stride = 2),
#           torch.nn.BatchNorm1d(128),
#           nn.ReLU()
#         )
        
#         self.up1 = nn.ConvTranspose1d(144, n_classes, 2, stride = 2)
        
    
    

    def forward(self, x):
        
        x = torch.transpose(x, 1,2) # [batch, seq_len, n_features] -> [batch, n_features, seq_len]
        norm_inputs = self.bn(x)
        
#         x1 = self.down1(x)#512->256
#         x2 = self.down2(x1)#256->128
#         x3 = self.down3(x2)#128->64
#         x4 = self.down4(x3)#64->32
#         #print(x2.size())
#         x3u = self.up4(x4)#32->64
        
#         x3 = torch.cat([x3, x3u], axis = 1) #64
#         x2u = self.up3(x3) #64->128
#         #print(x2u.size())
#         x2 = torch.cat([x2, x2u], axis = 1) #128
#         x1u = self.up2(x2) #128->256
#         x1 = torch.cat([x1, x1u], axis = 1) #256
#         logits = self.up1(x1)
        
        x = self.conv1(norm_inputs)
        x = torch.cat([x, norm_inputs], axis = 1)
        x = self.conv2(x)
        x = torch.cat([x, norm_inputs], axis = 1)
        
        x = torch.transpose(x, 1,2) # [batch, n_features, seq_len] -> [batch, seq_len, n_features]
        hidden = self.lstm(x)[0]
        hidden = torch.transpose(hidden, 1,2) # [batch, seq_len, n_features] -> [batch, n_features, seq_len]
        logits = self.head(hidden)
                            
        return logits

