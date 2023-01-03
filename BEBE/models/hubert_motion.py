import yaml
import numpy as np
import torch
import torch.nn as nn
from BEBE.models.supervised_nn_utils import SupervisedBehaviorModel, BEHAVIOR_DATASET
import fairseq

from torch.utils.data.dataset import Dataset
import os
import tqdm
import pickle
from torch.utils.data import DataLoader, Subset
import torchmetrics
import tqdm
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
from BEBE.models.model_superclass import BehaviorModel
import pandas as pd
    
class hubert_motion(SupervisedBehaviorModel):
  def __init__(self, config):
    super(hubert_motion, self).__init__(config)
    
    self.unknown_label = config['metadata']['label_names'].index('unknown')
    
    ## Todo add blur etc 
    
    self.model_path = self.model_config['model_path']
    assert self.model_path is not None, 'Need to specify path to pretrained model checkpoint'
    
    self.model = HubertMotionClassifier(self.model_config['model_path'],
                                        self.n_classes, 
                                        unfreeze_encoder = self.model_config['unfreeze_encoder']).to(self.device)
    
    
    
    print(self.model)

    print('Model parameters:')
    print(self._count_parameters())
    
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
      proportions = train_dataset.get_class_proportions() # Record class proportions for loss function
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
          # # unfreeze after 10 epochs have passed
          if t >= 10:
            self.model.unfreeze_encoder(self.model_config['unfreeze_encoder'])
          else:
            self.model.unfreeze_encoder(0)
          
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
          # scheduler.step()

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
    
  def save(self):
    pass # not implemented for hubert_motion
    
    
class HubertMotionClassifier(nn.Module):
    def __init__(self, model_path, num_classes, unfreeze_encoder = 0, embeddings_dim=768, multi_label=False):
        super().__init__()

        models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([model_path])
        self.model = models[0]
        self.unfreeze_encoder(unfreeze_encoder)
        self.head = nn.Linear(in_features=embeddings_dim, out_features=num_classes)
        # self.gru = nn.GRU(embeddings_dim, 64, 1, bidirectional = True)
        # self.head = nn.Linear(in_features=128, out_features=num_classes)
            
    def unfreeze_encoder(self, unfreeze):
        if unfreeze == -1:
          self.model.requires_grad_(True) # not worrying about bn running stats
          self.model.feature_extractor.requires_grad_(False)
        elif unfreeze == 0:
          self.model.requires_grad_(False) # not worrying about bn running stats
        else:
          self.model.requires_grad_(False) # not worrying about bn running stats
          for layer in self.model.encoder.layers[-unfreeze:]:
            layer.requires_grad_(True)
      
    def forward(self, x):
        B, L, C = x.size()
        out = self.model.extract_features(x)[0]
        # out, _ = self.gru(out)
        logits = self.head(out)
        logits = torch.transpose(logits, -1, -2)
        logits = nn.functional.interpolate(logits, size=L, mode='nearest-exact')
        return logits
