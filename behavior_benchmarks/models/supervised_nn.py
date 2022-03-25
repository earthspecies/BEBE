import yaml
import numpy as np
import pickle
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchmetrics
from behavior_benchmarks.models.pytorch_dataloaders import BEHAVIOR_DATASET
import tqdm
from matplotlib import pyplot as plt

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

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
    ##
    
    cols_included_bool = [x in self.config['input_vars'] for x in self.metadata['clip_column_names']] 
    self.cols_included = [i for i, x in enumerate(cols_included_bool) if x]
    
    labels_bool = [x == 'label' for x in self.metadata['clip_column_names']]
    self.label_idx = [i for i, x in enumerate(labels_bool) if x][0] # int
    
    self.n_classes = len(self.metadata['label_names']) 
    self.n_features = len(self.cols_included)
    
    self.model = LSTM_Classifier(self.n_features, self.n_classes, self.hidden_size, self.num_layers).to(device)
    print(self.model)
  
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
      test_fps = self.config['test_data_latents_fp']
    else:
      train_fps = self.config['train_data_fp']
      test_fps = self.config['test_data_fp']
    
    train_data = [self.load_model_inputs(fp, read_latents = self.read_latents) for fp in train_fps]
    #train_data = np.concatenate(train_data, axis = 0)
    test_data = [self.load_model_inputs(fp, read_latents = self.read_latents) for fp in test_fps]
    #test_data = np.concatenate(test_data, axis = 0)
    
    train_labels = [self.load_labels(fp) for fp in train_fps]
    #train_labels = np.concatenate(train_labels, axis = 0)
    test_labels = [self.load_labels(fp) for fp in test_fps]
    #test_labels = np.concatenate(test_labels, axis = 0)
    
    train_dataset = BEHAVIOR_DATASET(train_data, train_labels, True, self.temporal_window_samples)
    train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers = 0)
    
    test_dataset = BEHAVIOR_DATASET(test_data, test_labels, False, self.temporal_window_samples)
    test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers = 0)
    
    loss_fn = nn.CrossEntropyLoss(ignore_index = self.unknown_label)
    optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay, amsgrad = True)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, self.scheduler_epochs_between_step, gamma=0.1, last_epoch=- 1, verbose=False)
    
    train_loss = []
    test_loss = []
    train_acc = []
    test_acc = []
    
    epochs = self.n_epochs
    for t in range(epochs):
        print(f"Epoch {t}\n-------------------------------")
        l, a = self.train_epoch(train_dataloader, loss_fn, optimizer)
        train_loss.append(l)
        train_acc.append(a)
        l, a = self.test_epoch(test_dataloader, loss_fn)
        test_loss.append(l)
        test_acc.append(a)
        scheduler.step()
    print("Done!")
    
    # Save training progress
    plt.plot(train_loss, label= 'train', marker = '.')
    plt.plot(test_loss, label = 'test', marker = '.')
    plt.title("Cross Entropy Loss")
    plt.xlabel('Epoch')
    plt.xticks(np.arange(len(train_loss)))
    plt.ylabel('Loss')
    loss_fp = os.path.join(self.config['output_dir'], 'loss.png')
    plt.savefig(loss_fp)
    plt.close()
    
    # Save training progress
    plt.plot(train_acc, label= 'train', marker = '.')
    plt.plot(test_acc, label = 'test', marker = '.')
    plt.title("Mean accuracy")
    plt.xlabel('Epoch')
    plt.xticks(np.arange(len(train_acc)))
    plt.ylabel('Accuracy')
    acc_fp = os.path.join(self.config['output_dir'], 'acc.png')
    plt.savefig(acc_fp)
    plt.close()
    
    ###
    
  def train_epoch(self, dataloader, loss_fn, optimizer):
    size = len(dataloader.dataset)
    self.model.train()
    # Use accuracy instead of f1, since torchmetrics doesn't handle masking for precision as we want it to
    acc_score = torchmetrics.Accuracy(num_classes = self.n_classes, average = 'macro', mdmc_average = 'global', ignore_index = self.unknown_label)
    train_loss = 0
    num_batches_seen = 0

    with tqdm.tqdm(dataloader, unit = "batch") as tepoch:
      for i, (X, y) in enumerate(tepoch):
        if i % self.downsizing_factor != 0 :
          continue
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
    
  def test_epoch(self, dataloader, loss_fn):
    size = len(dataloader.dataset)
    num_batches_seen = 0
    self.model.eval()
    test_loss = 0
    # Use accuracy instead of f1, since torchmetrics doesn't handle masking for precision as we want it to
    acc_score = torchmetrics.Accuracy(num_classes = self.n_classes, average = 'macro', mdmc_average = 'global', ignore_index = self.unknown_label)
    
    with torch.no_grad():
        for i, (X, y) in enumerate(dataloader):
            if i % self.downsizing_factor != 0:
              continue
            num_batches_seen += 1
            X, y = X.type('torch.FloatTensor').to(device), y.type('torch.LongTensor').to(device)
            pred = self.model(X)
            acc_score.update(pred.cpu(), y.cpu())
            test_loss += loss_fn(pred, y).item()
    test_loss /= num_batches_seen
    acc = acc_score.compute()
    print("Test loss: %f, Test accuracy: %f" % (test_loss, acc))
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
    def __init__(self, n_features, n_classes, hidden_size, num_layers):
        super(LSTM_Classifier, self).__init__()
        
        self.head = nn.Conv1d(2*hidden_size, n_classes, 1, padding = 'same')
        self.lstm = nn.LSTM(n_features, hidden_size, num_layers = num_layers, bidirectional = True, batch_first = True)
        self.bn = torch.nn.BatchNorm1d(n_features)

    def forward(self, x):
        
        x = torch.transpose(x, 1,2) # [batch, seq_len, n_features] -> [batch, n_features, seq_len]
        x = self.bn(x)
        x = torch.transpose(x, 1,2) # [batch, n_features, seq_len] -> [batch, seq_len, n_features]
        hidden = self.lstm(x)[0]
        hidden = torch.transpose(hidden, 1,2) # [batch, seq_len, n_features] -> [batch, n_features, seq_len]
        logits = self.head(hidden)
                            
        return logits

