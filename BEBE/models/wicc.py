import yaml
import numpy as np
import pickle
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchmetrics
from BEBE.models.wicc_utils import BEHAVIOR_DATASET
import tqdm
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
from BEBE.models.model_superclass import BehaviorModel
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
import scipy.special as special
import math
import pandas as pd
from pathlib import Path
from BEBE.applications.S4.S4 import S4


# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"

def _count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class wicc(BehaviorModel):
  def __init__(self, config):
    super(wicc, self).__init__(config)
    print(f"Using {device} device")
    
    ##
    self.downsizing_factor = self.model_config['downsizing_factor']
    self.lr = self.model_config['lr']
    self.weight_decay = self.model_config['weight_decay']
    self.n_epochs = self.model_config['n_epochs']
    self.hidden_size = self.model_config['hidden_size']
    self.n_s4_blocks = self.model_config['num_layers'] ## Total layers is num_layers * 3 tiers
    self.context_window_samples = self.model_config['context_window_samples']
    self.context_window_stride = self.model_config['context_window_stride']
    self.batch_size = self.model_config['batch_size']
    self.dropout = self.model_config['dropout']
    self.blur_scale = self.model_config['blur_scale']
    self.jitter_scale = self.model_config['jitter_scale']
    self.state_size = self.model_config['state_size']
    self.downsample_rate = self.model_config['downsample_rate']
    self.n_clusters = self.config['num_clusters']
    self.temporal_window_samples = self.model_config['temporal_window_samples']
    self.n_pseudolabels = self.model_config['n_pseudolabels'] if 'n_pseudolabels' in self.model_config else self.n_clusters // 2
    self.max_iter_gmm = self.model_config['max_iter_gmm']
    self.tau_init = self.model_config['tau_init']
    self.tau_decay_rate = self.model_config['tau_decay_rate']
    self.feature_expansion_factor = self.model_config['feature_expansion_factor']
    self.diversity_alpha = self.model_config['diversity_alpha']
    self.pseudolabel_dir = self.model_config['pseudolabel_dir']
    self.individualized_head = self.model_config['individualized_head']
    self.per_frame_model = self.model_config['per_frame_model']
    ##
    
    assert self.context_window_samples % 2 != 0, 'context window should be an odd number'
    
    # cols_included_bool = [x in self.config['input_vars'] for x in self.metadata['clip_column_names']] 
    # self.cols_included = [i for i, x in enumerate(cols_included_bool) if x]
    
    labels_bool = [x == 'label' for x in self.metadata['clip_column_names']]
    self.label_idx = [i for i, x in enumerate(labels_bool) if x][0] # int
    
    self.n_features = len(self.cols_included)
    
    if self.individualized_head:
      self.dim_individual_embedding = max(self.config['metadata']['individual_ids']) + 1 # individuals are numbered 0, 1,..., highest, but may omit some integers
    else:
      self.dim_individual_embedding = 1
    
    self.encoder =  Encoder(self.n_features,
                            self.n_clusters,
                            hidden_size = self.hidden_size,
                            state_size = self.state_size,
                            n_s4_blocks = self.n_s4_blocks,
                            downsample_rate = self.downsample_rate,
                            feature_expansion_factor = self.feature_expansion_factor,
                            dropout = self.dropout,
                            blur_scale = self.blur_scale,
                            jitter_scale = self.jitter_scale).to(device)
    
    self.decoder = Decoder(self.n_clusters,
                           self.n_pseudolabels,
                           self.context_window_samples,
                           self.dim_individual_embedding).to(device)
    
    self.context_generator = ContextGenerator(self.context_window_samples, self.context_window_stride).to(device)
    
    print('Encoder parameters:')
    print(_count_parameters(self.encoder))
  
  def load_pseudolabels(self, filename):
    
    filepath = os.path.join(self.pseudolabel_dir, filename)
    labels = np.load(filepath).astype(int)
    return labels
  
  def generate_pseudolabels(self):
    ## Generate pseudo-labels
    print("Training per-frame models to produce pseudo-labels")
    
    if self.read_latents:
      dev_fps = self.config['dev_data_latents_fp']
      raise NotImplementedError
    else:
      dev_fps = self.config['dev_data_fp']
    
    bics = []
    aics = []
    for target_individual_id in tqdm.tqdm(self.config['metadata']['individual_ids']):
      dev_data = []
      individual_fps = []
      for fp in dev_fps: # search for files associated with that target_individual
        clip_id = fp.split('/')[-1].split('.')[0]
        individual_id = self.config['metadata']['clip_id_to_individual_id'][clip_id]
        if individual_id == target_individual_id:
          individual_fps.append(fp)
          dev_data.append(self.load_model_inputs(fp, read_latents = self.read_latents))
          
      if len(dev_data) == 0:
        continue

      dev_data = np.concatenate(dev_data, axis = 0)
      
      if self.per_frame_model == 'gmm':
        per_frame_model = GaussianMixture(n_components = self.n_pseudolabels, verbose = 0, max_iter = self.max_iter_gmm, n_init = 1)
      elif self.per_frame_model == 'kmeans':
        # per-channel normalization
        mu = np.mean(dev_data, axis = 0, keepdims = True)
        sigma = np.std(dev_data, axis = 0, keepdims = True)
        dev_data = (dev_data - mu) / (sigma + 1e-6)
        
        per_frame_model = KMeans(n_clusters=self.n_pseudolabels, n_init=10, max_iter=self.max_iter_gmm, verbose=0)
      
      per_frame_model.fit(dev_data)
      
      if self.per_frame_model == 'gmm':
        bics.append(per_frame_model.bic(dev_data))
        aics.append(per_frame_model.aic(dev_data))
      
      self.pseudolabel_dir = os.path.join(self.config['output_dir'], 'pseudolabels')
      if not os.path.exists(self.pseudolabel_dir):
        os.makedirs(self.pseudolabel_dir)

      #print("Generating pseudo-labels for dev data")
      for fp in individual_fps:
        data = self.load_model_inputs(fp, read_latents = self.read_latents)
        pseudolabels = per_frame_model.predict(data)
        target_fn = str(Path(fp).stem) + '.npy'
        target = os.path.join(self.pseudolabel_dir, target_fn)
        np.save(target, pseudolabels)
        
    if self.per_frame_model == 'gmm':
      gmm_ms = {}    
      gmm_ms['bic'] = float(np.mean(bics))
      gmm_ms['aic'] = float(np.mean(aics))
      gmm_ms['n_pseudolabels'] = self.n_pseudolabels
      gmm_model_selection_fp = os.path.join(self.config['output_dir'], 'gmm_model_selection.yaml')
      with open(gmm_model_selection_fp, 'w') as file:
        yaml.dump(gmm_ms, file)
  
  def fit(self):
    
    if self.pseudolabel_dir is None:
      self.generate_pseudolabels()
    
    ## get data. assume stored in memory for now
    if self.read_latents:
      raise NotImplementedError
      dev_fps = self.config['dev_data_latents_fp']
    else:
      dev_fps = self.config['dev_data_fp']
    
    dev_data = [self.load_model_inputs(fp, read_latents = self.read_latents) for fp in dev_fps]
    dev_ids = [pd.read_csv(fp, delimiter = ',', header = None).values[:, -2] for fp in dev_fps]
    # dev_ids = [np.load(fp)[:, -2] for fp in dev_fps] # assumes individual id is in column -2
    
    ## Load up pseudo-labels
    
    dev_pseudolabels = [self.load_pseudolabels(str(Path(fp).stem) + '.npy') for fp in self.config['dev_file_ids']]
    
    dev_dataset = BEHAVIOR_DATASET(dev_data, dev_pseudolabels, dev_ids, True, self.temporal_window_samples, self.context_window_samples, self.context_window_stride, self.dim_individual_embedding)
    dev_dataloader = DataLoader(dev_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers = 0)
    
    loss_fn = nn.CrossEntropyLoss(ignore_index = -1)
    diversity_loss_fn = DiversityLoss(alpha = self.diversity_alpha, n_clusters = self.n_clusters)
    
    optimizer = torch.optim.Adam([{'params' : self.encoder.parameters(), 'weight_decay' : self.weight_decay}, {'params' : self.decoder.parameters()}], lr=self.lr, amsgrad = True)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.n_epochs, eta_min=0, last_epoch=- 1, verbose=False)
    
    dev_loss = []
    dev_predictions_loss = []
    dev_diversity_loss = []
    # test_loss = []
    dev_acc = []
    # test_acc = []
    learning_rates = []
    
    epochs = self.n_epochs
    for t in range(epochs):
        print(f"Epoch {t}\n-------------------------------")
        l, pl, dl, a = self.train_epoch(t, dev_dataloader, loss_fn, diversity_loss_fn, optimizer)
        dev_loss.append(l)
        dev_predictions_loss.append(pl)
        dev_diversity_loss.append(dl)
        dev_acc.append(a)
        # # l, a = self.test_epoch(t, test_dataloader, 
        # #                        loss_fn_no_reduce, 
        # #                        name = "Test", 
        # #                        loss_denom = num_examples_test* self.temporal_window_samples * self.context_window_samples)
        # test_loss.append(l)
        # test_acc.append(a)
        
        learning_rates.append(optimizer.param_groups[0]["lr"])
        scheduler.step()
      
    print("Done!")
    
    ## Save training progress
    
    # Loss
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    
    ax.plot(dev_loss, label= 'total_loss_train', marker = '.')
    ax.plot(dev_predictions_loss, label = 'predictions_loss_train', marker = '.')
    ax.plot(dev_diversity_loss, label = 'diversity_loss_train', marker = '.')
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

    # Accuracy
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    ax.plot(dev_acc, label= 'dev', marker = '.')
    # ax.plot(test_acc, label = 'test', marker = '.')
    ax.legend()
    ax.set_title("Mean accuracy")
    ax.set_xlabel('Epoch')
    major_tick_spacing = max(1, len(dev_acc) // 10)
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
    
    # Cluster visualizations
    #w = self.decoder.weight.data.cpu().view((self.n_pseudolabels, self.context_window_samples, -1, self.n_clusters)).numpy() 
    w = self.decoder.weight.data.cpu().view((self.n_clusters, self.n_pseudolabels, self.context_window_samples, self.dim_individual_embedding)).numpy() 
    w = np.transpose(w, (1, 2, 3, 0))
    
    
    dev_individual_ids = []
    for fp in self.config['metadata']['dev_clip_ids']: # search for files associated with that target_individual
        clip_id = fp.split('/')[-1].split('.')[0]
        individual_id = self.config['metadata']['clip_id_to_individual_id'][clip_id]
        dev_individual_ids.append(individual_id)

    dev_individual_ids = sorted(set(dev_individual_ids))
    fig = plt.figure(figsize=(15, 15))
    ax = plt.subplot(111)
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)
    ax.set_title("Visualization of cluster meaning for different individuals")

    n_rows = min(5, self.n_clusters)
    n_cols = min(5, len(dev_individual_ids))

    for i in range(n_rows):
        for k in range(n_cols):
            axs = fig.add_subplot(n_rows, n_cols, 1 + n_cols * i + k)
            if self.individualized_head:
              ind = dev_individual_ids[k]
            else:
              ind = 0
            x = w[:, :, ind, i]
            # softmax over pseudolabels
            x = special.softmax(x, axis = 0) 
            for j in range(self.n_pseudolabels):
                axs.plot(x[j, :])
            axs.tick_params(
                axis='x',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom=False,      # ticks along the bottom edge are off
                top=False,         # ticks along the top edge are off
                labelbottom=False) # labels along the bottom edge are off

    ax.set_xlabel('Each column is a different individual; subplot x axis represents time')
    ax.set_ylabel('Each row is a different cluster; subplot y axis represents pseudolabel probability')
    clustervis_fp = os.path.join(self.config['output_dir'], 'wicc_clusters.png')
    fig.savefig(clustervis_fp)
    plt.close()
    
    # Model selection criteria
    model_selection = {}
    model_selection['final_acc'] = float(dev_acc[-1].item())
    model_selection['final_prediction_loss'] = dev_predictions_loss[-1]
    model_selection['final_diversity_loss'] = dev_diversity_loss[-1]
    model_selection['final_loss'] = dev_loss[-1]
    
    model_selection_fp = os.path.join(self.config['output_dir'], 'model_selection.yaml')
    with open(model_selection_fp, 'w') as file:
      yaml.dump(model_selection, file)
    
    
    
  def train_epoch(self, t, dataloader, loss_fn, diversity_loss_fn, optimizer):
    size = len(dataloader.dataset)
    self.encoder.train()
    gumbel_tau = self.tau_init * (self.tau_decay_rate ** t)
    acc_score = torchmetrics.Accuracy(mdmc_average = 'global').to(device)
    train_loss = 0
    train_predictions_loss = 0
    train_diversity_loss = 0
    num_batches_seen = 0
    
    num_batches_todo = 1 + len(dataloader) // self.downsizing_factor
    with tqdm.tqdm(dataloader, unit = "batch", total = num_batches_todo) as tepoch:
      for i, (X, y, individual_id) in enumerate(tepoch):
        if i == num_batches_todo :
          break
        num_batches_seen += 1
        X, y = X.to(device = device, dtype = torch.float), y.to(device = device, dtype = torch.float)
        
        # use a 1d convolution to generate context labels
        #y : [batch, temporal_window + padding]
        y = torch.unsqueeze(y, 1) # -> [batch, 1, temporal_window + padding]
        y = self.context_generator(y) # -> [batch, context_window, temporal_window]
        y = torch.transpose(y, 1,2) # -> [batch, temporal_window, context_window]
        y = y.to(torch.long)
        #
        individual_id = individual_id.to(device = device, dtype = torch.float)
        
        # Compute prediction error
        latent_logits = self.encoder(X)
        diversity_loss = diversity_loss_fn(latent_logits)
        q = torch.nn.functional.gumbel_softmax(latent_logits, tau=gumbel_tau, hard=True, dim=- 1) # [batch, seq_len, n_clusters]
        logits = self.decoder(q, individual_id)
        predictions_loss = loss_fn(logits, y) 
        loss = predictions_loss + diversity_loss
        train_loss += loss.item()
        train_predictions_loss += predictions_loss.item()
        train_diversity_loss += diversity_loss.item()
        
        labels_adjusted = y
        labels_adjusted = torch.maximum(labels_adjusted, torch.zeros_like(labels_adjusted)) # torchmetrics doesn't handle -1 labels so we treat them as gmm cluster number 0. introduces small error
        acc_score.update(logits, labels_adjusted)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        
        optimizer.step()
        loss_str = "%2.2f" % loss.item()
        tepoch.set_postfix(loss=loss_str)
        
    acc = acc_score.compute()
    # acc = 0.
    train_predictions_loss = train_predictions_loss / num_batches_seen
    train_diversity_loss = train_diversity_loss / num_batches_seen
    train_loss = train_loss / num_batches_seen
    print("Train loss: %f, Prediction loss %f, Diversity loss %f, Train accuracy: %f, Temperature: %f" % (train_loss, train_predictions_loss, train_diversity_loss, acc, gumbel_tau))
    return train_loss, train_predictions_loss, train_diversity_loss, acc
    
  def save(self):
    target_fp = os.path.join(self.config['final_model_dir'], "final_model.pickle")
    with open(target_fp, 'wb') as f:
      pickle.dump(self, f)
  
  
  def predict(self, data):
      self.encoder.eval()
      alldata= data

      predslist = []
      pred_len = self.temporal_window_samples

      current_start_step = 0
      for i in range(0, np.shape(alldata)[0] - 2* pred_len, pred_len):
        current_start_step = i
        data = alldata[current_start_step:current_start_step+pred_len, :] # window

        with torch.no_grad():
          data = np.expand_dims(data, axis =0)
          data = torch.from_numpy(data).type('torch.FloatTensor').to(device)
          preds = self.encoder(data)
          preds = preds.cpu().detach().numpy()
          preds = np.squeeze(preds, axis = 0)
          preds = np.argmax(preds, axis = -1).astype(np.uint8)

          predslist.append(preds)

      current_start_step += pred_len
      data = alldata[current_start_step:, :] # lump the last couple windows together; avoids tiny remainder windows
      with torch.no_grad():
        data = np.expand_dims(data, axis =0)
        data = torch.from_numpy(data).type('torch.FloatTensor').to(device)
        preds = self.encoder(data)
        preds = preds.cpu().detach().numpy()
        preds = np.squeeze(preds, axis = 0)
        preds = np.argmax(preds, axis = -1).astype(np.uint8)

        predslist.append(preds)

      preds = np.concatenate(predslist)
      return preds, None  
      
class S4Block(nn.Module):
    def __init__(self, H, N, dropout= 0.):
      super(S4Block, self).__init__()
      self.ln1 = nn.LayerNorm(H)
      self.s4 = S4(H, d_state = N, bidirectional = True, dropout = dropout, transposed = False)
      self.ln2 = nn.LayerNorm(H)
      self.linear2 = nn.Linear(H, 2*H)
      self.linear3 = nn.Linear(2*H, H)
      
    def forward(self, x):
      y = x
      x = self.ln1(x)
      x = self.s4(x)[0]
      x = y+ x
      
      y = x
      x = self.ln2(x)
      x = self.linear2(x)
      x = nn.functional.gelu(x)
      x = self.linear3(x)
      x = y+ x
      return x

class Encoder(nn.Module):
    def __init__(self, n_features, n_clusters, hidden_size, state_size, n_s4_blocks, downsample_rate, feature_expansion_factor, dropout, blur_scale = 0, jitter_scale = 0):
        super(Encoder, self).__init__()
        self.blur_scale = blur_scale
        self.jitter_scale = jitter_scale
        
        self.embedding = nn.Linear(n_features, hidden_size)
        
        self.bn = nn.BatchNorm1d(n_features)
        self.downsample_rate = downsample_rate
        self.down1 = nn.Conv1d(hidden_size, feature_expansion_factor * hidden_size, self.downsample_rate, stride = self.downsample_rate)
        self.down2 = nn.Conv1d(feature_expansion_factor * hidden_size, (feature_expansion_factor ** 2) * hidden_size, self.downsample_rate, stride = self.downsample_rate)
        
        self.s4_blocks_1 = nn.ModuleList([S4Block(hidden_size, state_size, dropout = dropout) for i in range(n_s4_blocks)])
        self.s4_blocks_2 = nn.ModuleList([S4Block(feature_expansion_factor * hidden_size, state_size, dropout = dropout) for i in range(n_s4_blocks)])
        self.s4_blocks_3 = nn.ModuleList([S4Block((feature_expansion_factor ** 2) * hidden_size, state_size, dropout = dropout) for i in range(n_s4_blocks)])
        self.head = nn.Conv1d((feature_expansion_factor ** 2) * hidden_size, n_clusters, 1, padding = 'same')
        
    def forward(self, x):
        seq_len = x.size()[-2]
        
        x = torch.transpose(x, -1, -2)
        x = self.bn(x)
        x = torch.transpose(x, -1, -2)
        
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
        
        x = self.embedding(x)
        
        for block in self.s4_blocks_1:
          x = block(x)
          
        x = torch.transpose(x, -1, -2)
        x = self.down1(x)
        x = torch.transpose(x, -1, -2)
        
        for block in self.s4_blocks_2:
          x = block(x)
          
        x = torch.transpose(x, -1, -2)
        x = self.down2(x)
        x = torch.transpose(x, -1, -2)
        
        for block in self.s4_blocks_3:
          x = block(x)
        
        x = torch.transpose(x, 1,2) # [batch, seq_len, n_features] -> [batch, n_features, seq_len]
        logits = self.head(x) # -> [batch, n_clusters, seq_len]
        
        logits = nn.functional.interpolate(logits, size=seq_len, mode='nearest-exact')
        logits = torch.transpose(logits, 1,2) # -> [batch, seq_len, n_clusters]
        return logits  

class Decoder(nn.Module):
  # Linear [batch, seq_len, n_clusters] (one-hot representation) -> [batch, n_pseudolabels, seq_len, context_window]
  def __init__(self, n_clusters, n_pseudolabels, context_window_samples, dim_individual_embedding):
      super(Decoder, self).__init__()
      self.prediction_head = nn.Linear(n_clusters, n_pseudolabels * context_window_samples * dim_individual_embedding, bias = False)
      self.n_pseudolabels = n_pseudolabels
      self.context_window_samples = context_window_samples
      self.dim_individual_embedding = dim_individual_embedding
      self.n_clusters = n_clusters
      self.weight = nn.parameter.Parameter(torch.empty((1, n_clusters, n_pseudolabels *context_window_samples, dim_individual_embedding))) # weight is [1, n_clusters, n_pseudolabels * context_window_samples, dim_individual_embedding]
      self.reset_parameters()
      
  def reset_parameters(self):
      # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
      # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
      # https://github.com/pytorch/pytorch/issues/57109
      nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
      
  def forward(self, q, individual_id):
      # q: one hot [batch, seq_len, n_clusters]
      # individual_id: one_hot [batch, dim_individual_embedding]
      
      # stack weights to match batch
      size = q.size()
      batch_size = size[0]
      seq_len = size[1]
      
      weight = self.weight.expand(batch_size, self.n_clusters, self.n_pseudolabels * self.context_window_samples, self.dim_individual_embedding)
      
      # unsqueeze so individual id is shape [batch, 1,1,dim_individual_embedding]
      individual_id = torch.unsqueeze(individual_id, 1)
      individual_id = torch.unsqueeze(individual_id, 1) 
      
      # grab weight only for relevant individual
      weight = weight * individual_id # ind_weights
      weight = torch.sum(weight, -1)# weight is shape [batch, n_clusters, n_pseudolabels * context_window_samples]
      logits = torch.bmm(q, weight) # logits is shape [batch, seq_len, n_pseudolabels *context_window_samples]
      
      # -> [batch, seq_len, n_pseudolabels, context_window]
      logits = logits.view(batch_size, seq_len, self.n_pseudolabels, self.context_window_samples)
      
      # -> [batch, n_pseudolabels, seq_len, context_window]
      logits = torch.transpose(logits, 1, 2)
      
      return logits

class ContextGenerator(nn.Module):
    # [batch, 1, padded_temporal_window] -> [batch, context_window, temporal_window]
    def __init__(self, context_window_samples, context_window_stride):
      super(ContextGenerator, self).__init__()
      self.generator_kernel = torch.eye(context_window_samples, dtype = torch.float, device = device) #
      self.generator_kernel = torch.unsqueeze(self.generator_kernel, 1) #[context_window, 1, context_window]
      self.context_window_stride = context_window_stride
      
    def forward(self, labels):
      return nn.functional.conv1d(labels, self.generator_kernel, dilation = self.context_window_stride)

class DiversityLoss(nn.Module):
    # Maximize label entropy across batches.
    # Following wav2vec 2.0
    def __init__(self, alpha, n_clusters):
        super(DiversityLoss, self).__init__()
        self.alpha = alpha
        self.normalizing_factor = 1. / n_clusters

    def forward(self, x):
        #following fairseq implementation, maximise perplexity 
        probs = nn.functional.softmax(x, dim = -1)
        avg_probs = torch.mean(probs, dim = [0, 1]) # -> [n_clusters]
        d = self.alpha *(1 - self.normalizing_factor * torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-7))))        
        return d
