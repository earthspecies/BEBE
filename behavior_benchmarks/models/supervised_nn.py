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

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"


def _count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class supervised_nn():
  def __init__(self, config):
    print(f"Using {device} device")
    self.config = config
    self.read_latents = config['read_latents']
    self.model_config = config['supervised_nn_config']
    self.metadata = config['metadata']
    self.unknown_label = config['metadata']['label_names'].index('unknown')
    
    ##
    self.downsizing_factor = self.model_config['downsizing_factor']
    self.lr = self.model_config['lr']
    self.weight_decay = self.model_config['weight_decay']
    self.n_epochs = self.model_config['n_epochs']
    self.hidden_size = self.model_config['hidden_size']
    self.num_layers_lstm = self.model_config['num_layers']
    self.temporal_window_samples = self.model_config['temporal_window_samples']
    self.batch_size = self.model_config['batch_size']
    self.dropout = self.model_config['dropout']
    self.blur_scale = self.model_config['blur_scale']
    self.jitter_scale = self.model_config['jitter_scale']
    self.rescale_param = self.model_config['rescale_param']
    self.conv_stack_depth = self.model_config['conv_stack_depth']
    self.sparse_annotations = self.model_config['sparse_annotations']
    ##
    
    cols_included_bool = [x in self.config['input_vars'] for x in self.metadata['clip_column_names']] 
    self.cols_included = [i for i, x in enumerate(cols_included_bool) if x]
    
    labels_bool = [x == 'label' for x in self.metadata['clip_column_names']]
    self.label_idx = [i for i, x in enumerate(labels_bool) if x][0] # int
    
    self.n_classes = len(self.metadata['label_names']) 
    self.n_features = len(self.cols_included)
    
    self.model = LSTM_Classifier(self.n_features, self.n_classes, self.hidden_size, self.num_layers_lstm, self.dropout, self.conv_stack_depth).to(device)
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
    
    train_dataset = BEHAVIOR_DATASET(train_data, train_labels, True, self.temporal_window_samples, rescale_param = self.rescale_param)
    if self.sparse_annotations:
      indices_to_keep = train_dataset.get_annotated_windows()
      train_dataset = Subset(train_dataset, indices_to_keep)  
      print("Number windowed train examples after subselecting: %d" % len(train_dataset))
    train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers = 0)
    
    val_dataset = BEHAVIOR_DATASET(val_data, val_labels, False, self.temporal_window_samples)
    if self.sparse_annotations:
      indices_to_keep = val_dataset.get_annotated_windows()
      val_dataset = Subset(val_dataset, indices_to_keep) 
      
    num_examples_val = len(list(range(0, len(val_dataset), self.downsizing_factor)))
    val_dataset = Subset(val_dataset, list(range(0, len(val_dataset), self.downsizing_factor)))
    print("Number windowed val examples after subselecting: %d" % len(val_dataset))
    val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False, num_workers = 0)
    
    test_dataset = BEHAVIOR_DATASET(test_data, test_labels, False, self.temporal_window_samples)
    if self.sparse_annotations:
      indices_to_keep = test_dataset.get_annotated_windows()
      test_dataset = Subset(test_dataset, indices_to_keep)

    num_examples_test = len(list(range(0, len(test_dataset), self.downsizing_factor)))
    test_dataset = Subset(test_dataset, list(range(0, len(test_dataset), self.downsizing_factor)))
    print("Number windowed test examples after subselecting: %d" % len(test_dataset))
    test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False, num_workers = 0)
    
    loss_fn = nn.CrossEntropyLoss(ignore_index = self.unknown_label)
    loss_fn_no_reduce = nn.CrossEntropyLoss(ignore_index = self.unknown_label, reduction = 'sum')
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
    with torch.no_grad():
      data = np.expand_dims(data, axis =0)
      data = torch.from_numpy(data).type('torch.FloatTensor').to(device)
      preds = self.model(data)
      preds = preds.cpu().detach().numpy()
      preds = np.squeeze(preds, axis = 0)
      preds = np.argmax(preds, axis = 0).astype(np.uint8)
      #print(preds.dtype)
    return preds, None
    ###
  
  def predict_from_file(self, fp):
    inputs = self.load_model_inputs(fp, read_latents = self.read_latents)
    predictions, latents = self.predict(inputs)
    return predictions, latents

class LSTM_Classifier(nn.Module):
    def __init__(self, n_features, n_classes, hidden_size, num_layers_lstm, dropout, conv_stack_depth, blur_scale = 0, jitter_scale = 0):
        super(LSTM_Classifier, self).__init__()
        self.blur_scale = blur_scale
        self.jitter_scale = jitter_scale
        
        self.head = nn.Conv1d(2*hidden_size, n_classes, 1, padding = 'same')
        self.lstm = nn.LSTM(n_features + 64, hidden_size, num_layers = num_layers_lstm, bidirectional = True, batch_first = True, dropout = dropout)
        
        self.conv_stack = [_conv_block(n_features, 64, 64, 7)]
        for i in range(conv_stack_depth - 1):
          self.conv_stack.append(_conv_block(64+n_features, 64 + n_features, 64 + n_features, 3)) 
        self.conv_stack = nn.ModuleList(self.conv_stack)
        
        self.bn = torch.nn.BatchNorm1d(n_features)
        self.dropout = nn.Dropout(p = 0.01)

    def forward(self, x):
        
        x = torch.transpose(x, 1,2) # [batch, seq_len, n_features] -> [batch, n_features, seq_len]
        norm_inputs = self.bn(x)
        
        if self.training:
          # Perform augmentations to normalized data
          size = norm_inputs.size()
          blur = self.blur_scale * torch.randn(size, device = norm_inputs.device)
          jitter = self.jitter_scale *torch.randn((size[0], size[1], 1), device = norm_inputs.device)
          norm_inputs = norm_inputs + blur + jitter 
        
        x = self.conv_stack[0](norm_inputs)
        x = torch.cat([x, norm_inputs], axis = 1)
      
        for layer in self.conv_stack[1:]:
          x = layer(x) + x
        
        x = torch.transpose(x, 1,2) # [batch, n_features, seq_len] -> [batch, seq_len, n_features]
        hidden = self.lstm(x)[0]
        hidden = torch.transpose(hidden, 1,2) # [batch, seq_len, n_features] -> [batch, n_features, seq_len]
        logits = self.head(hidden)
                            
        return logits
      
def _conv_block(in_dims, hidden_dims, out_dims, kernel_width):
  block = nn.Sequential(
    nn.Conv1d(in_dims,hidden_dims, kernel_width, bias= False, padding = 'same'),
    torch.nn.BatchNorm1d(hidden_dims),
    nn.ReLU(),
    nn.Conv1d(hidden_dims,out_dims,kernel_width, bias= False, padding = 'same'),
    torch.nn.BatchNorm1d(out_dims),
    nn.ReLU()
  )
  return block

