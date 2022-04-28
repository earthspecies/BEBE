import yaml
import numpy as np
import pickle
import os
import torch
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
import torchmetrics
from behavior_benchmarks.models.vqcpc_utils import WarmupScheduler, CPCDataset, Encoder, CPCLoss
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator

from itertools import chain
import torch.optim as optim
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

def _count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class vq_cpc():
  def __init__(self, config):
    # Get cpu or gpu device for training.
    print(f"Using {device} device")
    self.config = config
    self.read_latents = config['read_latents']
    self.model_config = config['vq_cpc_config']
    self.metadata = config['metadata']
    self.unknown_label = config['metadata']['label_names'].index('unknown')
    
    ##
    self.downsizing_factor = self.model_config['downsizing_factor']
    self.lr = self.model_config['lr']
    # self.weight_decay = self.model_config['weight_decay']
    self.n_epochs = self.model_config['n_epochs']
    self.conv_stack_hidden_size = self.model_config['conv_stack_hidden_size']
    self.temporal_window_samples = self.model_config['temporal_window_samples']
    self.predict_proportion = self.model_config['predict_proportion']
    self.encoder_kernel_width = self.model_config['encoder_kernel_width']
    self.batch_size =self.model_config['batch_size']
    # self.dropout = self.model_config['dropout']
    # self.blur_scale = self.model_config['blur_scale']
    # self.jitter_scale = self.model_config['jitter_scale']
    # self.rescale_param = self.model_config['rescale_param']
    self.conv_stack_depth = self.model_config['conv_stack_depth']
    self.z_dim = self.model_config['z_dim']
    self.c_dim = self.model_config['c_dim']
    self.warmup_epochs = self.model_config['warmup_epochs']
    self.initial_lr = self.model_config['initial_lr']
    self.blur_scale = self.model_config['blur_scale']
    self.jitter_scale = self.model_config['jitter_scale']
    self.pooling_factor = self.model_config['pooling_factor']
    # ##
    
    cols_included_bool = [x in self.config['input_vars'] for x in self.metadata['clip_column_names']] 
    self.cols_included = [i for i, x in enumerate(cols_included_bool) if x]
    
    labels_bool = [x == 'label' for x in self.metadata['clip_column_names']]
    self.label_idx = [i for i, x in enumerate(labels_bool) if x][0] # int
    
    self.n_classes = len(self.metadata['label_names']) 
    self.n_features = len(self.cols_included)
    
    ##
    self.model = Encoder(self.n_features, self.conv_stack_hidden_size, self.config['num_clusters'], self.z_dim, self.c_dim, self.encoder_kernel_width, self.conv_stack_depth, blur_scale = self.blur_scale, jitter_scale = self.jitter_scale)
    self.cpc = CPCLoss(1, self.batch_size, int(self.temporal_window_samples * self.predict_proportion / self.pooling_factor), 8, self.z_dim, self.c_dim)
    self.model.to(device)
    self.cpc.to(device)
    ##
    
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
      dev_fps = self.config['dev_data_latents_fp']
      test_fps = self.config['test_data_latents_fp']
    else:
      dev_fps = self.config['dev_data_fp']
      test_fps = self.config['test_data_fp']
    
    dev_data = [self.load_model_inputs(fp, read_latents = self.read_latents) for fp in dev_fps]
    test_data = [self.load_model_inputs(fp, read_latents = self.read_latents) for fp in test_fps]
    
    dev_labels = [self.load_labels(fp) for fp in dev_fps]
    test_labels = [self.load_labels(fp) for fp in test_fps]
    
    ###
    dev_dataset = CPCDataset(dev_data, True, self.temporal_window_samples)
    dev_dataloader = DataLoader(dev_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers = 0)
    
    test_dataset = CPCDataset(test_data, False, self.temporal_window_samples)
    test_dataset = Subset(test_dataset, list(range(0, len(test_dataset), self.downsizing_factor)))
    print("Number windowed test examples after subselecting: %d" % len(test_dataset))
    test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers = 0)
    
    loss_fn = nn.CrossEntropyLoss(ignore_index = self.unknown_label)
    loss_fn_no_reduce = nn.CrossEntropyLoss(ignore_index = self.unknown_label, reduction = 'sum')
    #optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay, amsgrad = True)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.n_epochs, eta_min=0, last_epoch=- 1, verbose=False)
    
    optimizer = optim.Adam(
        chain(self.model.parameters(), self.cpc.parameters()),
        lr=self.initial_lr)
    
    scheduler = WarmupScheduler(
        optimizer,
        warmup_epochs=self.warmup_epochs,
        initial_lr=self.initial_lr,
        max_lr=self.lr,
        milestones= [20000],
        gamma=0.25)
    
    train_cpc_loss = []
    train_vq_loss = []
    train_perplexity = []
    train_accuracy = []
    
    test_cpc_loss = []
    test_vq_loss = []
    test_perplexity = []
    test_accuracy = []
    
    learning_rates = []
    
    for epoch in range(self.n_epochs):
        print(f"Epoch {epoch}\n-------------------------------")
        cpc_loss, vq_loss, perplexity, accuracy = self.train_epoch(dev_dataloader, loss_fn, optimizer)
        train_cpc_loss.append(cpc_loss)
        train_vq_loss.append(vq_loss)
        train_perplexity.append(perplexity)
        train_accuracy.append(accuracy)
        
        cpc_loss, vq_loss, perplexity, accuracy = self.test_epoch(test_dataloader, loss_fn, optimizer)
        test_cpc_loss.append(cpc_loss)
        test_vq_loss.append(vq_loss)
        test_perplexity.append(perplexity)
        test_accuracy.append(accuracy)
        
        learning_rates.append(optimizer.param_groups[0]["lr"])
        scheduler.step()
        

    print("Done!")
    
    ## Save training progress
    
    # Loss
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    
    ax.plot(train_cpc_loss, label= 'train_cpc', marker = '.')
    ax.plot(test_cpc_loss, label = 'test_cpc', marker = '.')
    ax.plot(train_vq_loss, label= 'train_vq', marker = '.')
    ax.plot(test_vq_loss, label = 'test_vq', marker = '.')
    ax.legend()
    ax.set_title("Loss")
    ax.set_xlabel('Epoch')
    
    major_tick_spacing = max(1, len(train_cpc_loss) // 10)
    ax.xaxis.set_major_locator(MultipleLocator(major_tick_spacing))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.set_ylabel('Loss')
    loss_fp = os.path.join(self.config['output_dir'], 'loss.png')
    fig.savefig(loss_fp)
    plt.close()

    # Accuracy
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    ax.plot(train_accuracy, label= 'train', marker = '.')
    ax.plot(test_accuracy, label = 'test', marker = '.')
    ax.legend()
    ax.set_title("Mean accuracy")
    ax.set_xlabel('Epoch')
    major_tick_spacing = max(1, len(train_accuracy) // 10)
    ax.xaxis.set_major_locator(MultipleLocator(major_tick_spacing))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.set_ylabel('Accuracy')
    acc_fp = os.path.join(self.config['output_dir'], 'acc.png')
    fig.savefig(acc_fp)
    plt.close()
    
    # Perplexity
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    ax.plot(train_perplexity, label= 'train', marker = '.')
    ax.plot(test_perplexity, label = 'test', marker = '.')
    ax.legend()
    ax.set_title("Perplexity")
    ax.set_xlabel('Epoch')
    major_tick_spacing = max(1, len(train_perplexity) // 10)
    ax.xaxis.set_major_locator(MultipleLocator(major_tick_spacing))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.set_ylabel('Perplexity')
    perp_fp = os.path.join(self.config['output_dir'], 'perplexity.png')
    fig.savefig(perp_fp)
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
    ##
    
  def train_epoch(self, dataloader, loss_fn, optimizer):
    size = len(dataloader.dataset)
    self.model.train()
    num_batches_seen = 0
    
    num_batches_todo = 1 + len(dataloader) // self.downsizing_factor
    
    vq_losses = []
    perplexities = []
    cpc_losses = []
    accuracies = []
    losses = []
    
    with tqdm(dataloader, unit = "batch", total = num_batches_todo) as tepoch:
      for i, X in enumerate(tepoch):
        if i == num_batches_todo :
          break
        X = X.type('torch.FloatTensor').to(device)
        #X = X.view(cfg.training.n_speakers_per_batch *cfg.training.n_utterances_per_speaker,cfg.preprocessing.n_mels, -1)

        optimizer.zero_grad()

        z, c, vq_loss, perplexity = self.model(X)
        cpc_loss, accuracy = self.cpc(z, c)
        loss = cpc_loss + vq_loss
        
        vq_losses.append(vq_loss.item())
        perplexities.append(perplexity.item())
        cpc_losses.append(cpc_loss.item())
        accuracies.extend(accuracy)
        losses.append(loss.item())

        loss.backward()
        optimizer.step()
        num_batches_seen += 1
        
        loss_str = "%2.2f" % loss.item()
        tepoch.set_postfix(loss=loss_str)
        
    
    train_loss = np.mean(losses)
    vq_loss = np.mean(vq_losses)
    perplexity = np.mean(perplexities)
    cpc_loss = np.mean(cpc_losses)
    accuracy = np.mean(accuracies)
    print("Train loss: %f, Train accuracy: %f" % (train_loss, accuracy))
    return cpc_loss, vq_loss, perplexity, accuracy
    
  def test_epoch(self, dataloader, loss_fn, name = "Test", loss_denom = 0):
    self.model.eval()
    
    vq_losses = []
    perplexities = []
    cpc_losses = []
    accuracies = []
    losses = []
    
    with torch.no_grad():
      for i, X in enumerate(dataloader):
        
        X = X.type('torch.FloatTensor').to(device)
        #X = X.view(cfg.training.n_speakers_per_batch *cfg.training.n_utterances_per_speaker,cfg.preprocessing.n_mels, -1)

        z, c, vq_loss, perplexity = self.model(X)
        cpc_loss, accuracy = self.cpc(z, c)
        loss = cpc_loss + vq_loss
        
        vq_losses.append(vq_loss.item())
        perplexities.append(perplexity.item())
        cpc_losses.append(cpc_loss.item())
        accuracies.extend(accuracy)
        losses.append(loss.item())

        
    
    test_loss = np.mean(losses)
    vq_loss = np.mean(vq_losses)
    perplexity = np.mean(perplexities)
    cpc_loss = np.mean(cpc_losses)
    accuracy = np.mean(accuracies)
    print("Test loss: %f, Test accuracy: %f" % (test_loss, accuracy))
    return cpc_loss, vq_loss, perplexity, accuracy
    
  def save(self):
    target_fp = os.path.join(self.config['final_model_dir'], "final_model.pickle")
    with open(target_fp, 'wb') as f:
      pickle.dump(self, f)
  
  def predict(self, data):
    ###
    self.model.eval()
    with torch.no_grad():
      data = np.expand_dims(data, axis =0)
      
      
      
      X = torch.from_numpy(data).type('torch.FloatTensor').to(device)
      latents, _, preds = self.model.encode(X)
      
      
      # data = torch.from_numpy(data).type('torch.FloatTensor').to(device)
      # latents, _, preds = self.model.encode(data)
      ####
      preds = preds.cpu().detach().numpy()
      latents = latents.cpu().detach().numpy()
      # print("preds shape:")
      # print(np.shape(preds))
      # print("latents shape:")
      # print(np.shape(latents))
      preds = np.squeeze(preds, axis = 0).astype(np.uint8)
      latents = np.squeeze(latents, axis = 0)
      #preds = np.argmax(preds, axis = 0).astype(np.uint8)
      #print(preds.dtype)
    return preds, latents
    ###
  
  def predict_from_file(self, fp):
    inputs = self.load_model_inputs(fp, read_latents = self.read_latents)
    predictions, latents = self.predict(inputs)
    return predictions, latents

