import torch
from torch.utils.data.dataset import Dataset
import numpy as np
import os
import tqdm
import yaml
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchmetrics
import tqdm
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
from BEBE.models.model_superclass import BehaviorModel
import random
import pandas as pd
from BEBE.models.preprocess import static_acc_filter, normalize_acc_magnitude

    
class BEHAVIOR_DATASET(Dataset):
    def __init__(self, data, labels, train, temporal_window_samples, config, rescale_param = 0):
        self.temporal_window = temporal_window_samples
        self.rescale_param = rescale_param
        
        self.data = data # list of np arrays, each of shape [*, n_features] where * is the number of samples and varies between arrays
        self.labels = labels # list of np arrays, each of shape [*,] where * is the number of samples and varies between arrays
        
        label_names = config['metadata']['label_names']
        self.num_classes = len(label_names)
        self.unknown_idx = label_names.index('unknown')
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
        self.rng = np.random.default_rng(config['seed'])
        self.train = train
        
    def __len__(self):        
        return self.data_points - len(self.data) * self.temporal_window
      
    def get_class_proportions(self):
        all_labels = np.concatenate(self.labels)
        counts = []
        for i in range(self.num_classes):
          counts.append(len(all_labels[all_labels == i]))
        total_labels = sum(counts[:self.unknown_idx] + counts[self.unknown_idx + 1:])
        weights = np.array([x/total_labels for x in counts], dtype = 'float')
        return torch.from_numpy(weights).type(torch.FloatTensor)
      
    def get_annotated_windows(self):
        # Go through data, make a list of the indices of windows which actually have annotations.
        # Useful for speeding up supervised train time for sparsely labeled datasets.
        indices_of_annotated_windows = []
        print("Subselecting data to speed up training")
        for index in tqdm.tqdm(range(self.__len__())):
          clip_number = np.where(index >= self.data_start_indices)[0][-1] #which clip do I draw from?
          labels_item = self.labels[clip_number]
        
          start = index - self.data_start_indices[clip_number]
          end = start+ self.temporal_window
          
          if np.any(labels_item[start:end] != 0):
            indices_of_annotated_windows.append(index)
        return indices_of_annotated_windows

    def __getitem__(self, index):
        clip_number = np.where(index >= self.data_start_indices)[0][-1] #which clip do I draw from?
        
        data_item = self.data[clip_number]
        labels_item = self.labels[clip_number]
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
          labels_item = labels_item[start:end] 
        
        data_item = torch.from_numpy(data_item)
        labels_item = torch.from_numpy(labels_item)
            
        return data_item, labels_item

class SupervisedBehaviorModel(BehaviorModel):
  def __init__(self, config):
    super(SupervisedBehaviorModel, self).__init__(config)
    
    torch.manual_seed(self.config['seed'])
    random.seed(self.config['seed'])
    np.random.seed(self.config['seed'])
    
    # Get cpu or gpu device for training.
    self.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {self.device} device")
    
    ## General Training Parameters
    self.downsizing_factor = self.model_config['downsizing_factor']
    self.lr = self.model_config['lr']
    self.weight_decay = self.model_config['weight_decay']
    self.n_epochs = self.model_config['n_epochs']
    self.temporal_window_samples = self.model_config['temporal_window_samples']
    self.batch_size = self.model_config['batch_size']
    self.dropout = self.model_config['dropout']
    self.blur_scale = self.model_config['blur_scale']
    self.jitter_scale = self.model_config['jitter_scale']
    self.rescale_param = self.model_config['rescale_param']
    self.sparse_annotations = self.model_config['sparse_annotations']
    self.normalize = self.model_config['normalize']
    
    # Dataset Parameters
    self.unknown_label = config['metadata']['label_names'].index('unknown')
    labels_bool = [x == 'label' for x in self.metadata['clip_column_names']]
    self.label_idx = [i for i, x in enumerate(labels_bool) if x][0] # int
    self.n_classes = len(self.metadata['label_names']) 
    self.n_features = self.get_n_features()
    
    # Specify in subclass
    self.model = None
    
  def _count_parameters(self):
    return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
  
  def get_n_features(self):
    # gets number of input channels; this varies depending on static acc filtering hyperparameter
    train_fps = self.config['train_data_fp']
    x = self.load_model_inputs(train_fps[0])
    return np.shape(x)[1]
    
  def load_model_inputs(self, filepath):
    x = pd.read_csv(filepath, delimiter = ',', header = None).values[:, self.cols_included] #[n_samples, n_features]
    x = static_acc_filter(x, self.config)

    if self.normalize:
      x = normalize_acc_magnitude(x, self.config)

    return x
    
  def load_labels(self, filepath):
    labels = pd.read_csv(filepath, delimiter = ',', header = None).values[:, self.label_idx].astype(int)
    return labels 
    
  def fit(self):
    train_fps = self.config['train_data_fp']
    val_fps = self.config['val_data_fp']
    test_fps = self.config['test_data_fp']
    
    train_data = [self.load_model_inputs(fp) for fp in train_fps]
    val_data = [self.load_model_inputs(fp) for fp in val_fps]
    test_data = [self.load_model_inputs(fp) for fp in test_fps]
    
    train_labels = [self.load_labels(fp) for fp in train_fps]
    val_labels = [self.load_labels(fp) for fp in val_fps]
    test_labels = [self.load_labels(fp) for fp in test_fps]
    
    train_dataset = BEHAVIOR_DATASET(train_data, train_labels, True, self.temporal_window_samples, self.config, rescale_param = self.rescale_param)
    proportions = train_dataset.get_class_proportions() # Record class proportions for loss function
    if self.sparse_annotations:
      indices_to_keep = train_dataset.get_annotated_windows()
      train_dataset = Subset(train_dataset, indices_to_keep)  
      print("Number windowed train examples after subselecting: %d" % len(train_dataset))
    train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers = 0)
    
    if len(val_data) > 0:
      val_dataset = BEHAVIOR_DATASET(val_data, val_labels, False, self.temporal_window_samples, self.config)
      if self.sparse_annotations:
        indices_to_keep = val_dataset.get_annotated_windows()
        val_dataset = Subset(val_dataset, indices_to_keep) 

      num_examples_val = len(list(range(0, len(val_dataset), self.downsizing_factor)))
      val_dataset = Subset(val_dataset, list(range(0, len(val_dataset), self.downsizing_factor)))
      print("Number windowed val examples after subselecting: %d" % len(val_dataset))
      val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False, num_workers = 0)
      use_val = True
    else:
      use_val = False
    
    test_dataset = BEHAVIOR_DATASET(test_data, test_labels, False, self.temporal_window_samples, self.config)
    if self.sparse_annotations:
      indices_to_keep = test_dataset.get_annotated_windows()
      test_dataset = Subset(test_dataset, indices_to_keep)

    num_examples_test = len(list(range(0, len(test_dataset), self.downsizing_factor)))
    test_dataset = Subset(test_dataset, list(range(0, len(test_dataset), self.downsizing_factor)))
    print("Number windowed test examples after subselecting: %d" % len(test_dataset))
    test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False, num_workers = 0)
    
    # Loss function; reweight by class proportions
    weight = 1./ (proportions + 1e-6).to(self.device) 
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
        if use_val:
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
        X, y = X.type('torch.FloatTensor').to(self.device), y.type('torch.LongTensor').to(self.device)
        
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
            X, y = X.type('torch.FloatTensor').to(self.device), y.type('torch.LongTensor').to(self.device)
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
  
  def predict(self, alldata):
    self.model.eval()
    predslist = []
    pred_len = self.temporal_window_samples
    for i in range(0, np.shape(alldata)[0], pred_len):
      data = alldata[i:i+pred_len, :] # window to acommodate more hidden states
      if np.shape(data)[0]<pred_len:
        orig_len = np.shape(data)[0]
        pad = pred_len-np.shape(data)[0]
        data = np.pad(data, ((0,pad),(0,0)), mode = 'mean')
      else:
        pad = 0
      
      with torch.no_grad():
        data = np.expand_dims(data, axis =0)
        data = torch.from_numpy(data).type('torch.FloatTensor').to(self.device)
        preds = self.model(data)
        preds = preds.cpu().detach().numpy()
        preds = np.squeeze(preds, axis = 0)
        preds = np.argmax(preds, axis = 0).astype(np.uint8)
        if pad>0:
          preds = preds[:orig_len]
        predslist.append(preds)
      
    preds = np.concatenate(predslist)
    return preds, None  

