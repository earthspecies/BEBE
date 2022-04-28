import torch
from torch.utils.data.dataset import Dataset
import torch.nn as nn
import numpy as np
import os
import torch.optim as optim
from collections import Counter
import warnings

import torch.nn.functional as F
from torch.distributions import Categorical
import math

### Dataset

class CPCDataset(Dataset):
    def __init__(self, data, train, temporal_window_samples, individual_idx = -2):
        self.temporal_window = temporal_window_samples
        self.individual_idx = individual_idx
        
        self.data = data # list of np arrays, each of shape [*, n_features] where * is the number of samples and varies between arrays
        
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
        self.rng = np.random.default_rng()
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
      
# Model architecture
      
class Encoder(nn.Module):
    def __init__(self, in_channels, channels, n_embeddings, z_dim, c_dim, kernel_width, conv_stack_depth, blur_scale = 0, jitter_scale = 0, pooling_factor = 1):
        super(Encoder, self).__init__()
        # self.conv = nn.Conv1d(in_channels, channels, 4, 2, 1, bias=False)
        self.pool_size = pooling_factor
        
        self.blur_scale = blur_scale
        self.jitter_scale = jitter_scale
        
        self.codebook = VQEmbeddingEMA(n_embeddings, z_dim)
        self.rnn = nn.LSTM(z_dim, c_dim, batch_first=True)
        
        self.conv_stack = [_conv_block(in_channels, channels, channels-in_channels, kernel_width)]
        for i in range(conv_stack_depth - 1):
          self.conv_stack.append(_conv_block(channels, channels, channels, kernel_width)) 
        self.conv_stack = nn.ModuleList(self.conv_stack)
        #self.head = nn.Conv1d(channels, z_dim, 1, padding = 'same')
        
        pooling = nn.AvgPool1d(self.pool_size)
        self.head = nn.Sequential(pooling, nn.Conv1d(channels, z_dim, 1, padding = 'same'))
        
        self.bn = torch.nn.BatchNorm1d(in_channels)
        

    def encode(self, x):
        # z = self.conv(mel)
        # z = self.encoder(z.transpose(1, 2))
        
        x = torch.transpose(x, 1,2) # [batch, seq_len, n_features] -> [batch, n_features, seq_len]
        
        
        x_len_samples = x.size()[-1] 
        pad_length = self.pool_size - (x_len_samples % self.pool_size)
        
        x = nn.functional.pad(x, (0, pad_length))
        
        
        norm_inputs = self.bn(x)
        
        x = self.conv_stack[0](norm_inputs)
        x = torch.cat([x, norm_inputs], axis = 1)
      
        for layer in self.conv_stack[1:]:
          x = layer(x) + x
        
        x = self.head(x)
        z = torch.transpose(x, 1,2) #[batch, seq_len, n_features]
        
        z, indices = self.codebook.encode(z)
        
        
        c, _ = self.rnn(z)
        
        # upsample predictions and quantized latents for analysis
        upsampler = nn.Upsample(scale_factor=self.pool_size, mode='nearest')
        
        indices = torch.unsqueeze(indices, 1)
        indices = upsampler(indices.type('torch.FloatTensor'))
        indices = torch.squeeze(indices, 1)
        indices = indices[:, :x_len_samples].type('torch.LongTensor')
        
        z = torch.transpose(z, 1,2) #[batch, n_features, seq_len]
        z = upsampler(z)
        z = z[:, :, :x_len_samples]
        z = torch.transpose(z, 1,2) #[batch, seq_len, n_features]
        
        ##
        # upsampling of c not currently implemented
        ##
        
        return z, c, indices

    def forward(self, x):
        # z = self.conv(mel)
        # z = self.encoder(z.transpose(1, 2))
        
        x = torch.transpose(x, 1,2) # [batch, seq_len, n_features] -> [batch, n_features, seq_len]
        norm_inputs = self.bn(x)
        
        if self.training:
          # Perform augmentations to normalized data
          size = norm_inputs.size()
          blur = self.blur_scale * torch.randn(size, device = norm_inputs.device)
          jitter = self.jitter_scale *torch.randn((size[0], size[1], 1), device = norm_inputs.device)
          norm_inputs = norm_inputs + blur + jitter 
          
        x_len_samples = x.size()[-1] #int(x.size()[-1].item())
        pad_length = x_len_samples % self.pool_size
        x = nn.functional.pad(x, (0, pad_length))
        
        x = self.conv_stack[0](norm_inputs)
        x = torch.cat([x, norm_inputs], axis = 1)
      
        for layer in self.conv_stack[1:]:
          x = layer(x) + x
        
        x = self.head(x)
        z = torch.transpose(x, 1,2)
        z, loss, perplexity = self.codebook(z)       
        
        c, _ = self.rnn(z)
        return z, c, loss, perplexity

# Codebook

class VQEmbeddingEMA(nn.Module):
    def __init__(self, n_embeddings, embedding_dim, commitment_cost=0.25, decay=0.999, epsilon=1e-5):
        super(VQEmbeddingEMA, self).__init__()
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon

        init_bound = 1 / 512
        embedding = torch.Tensor(n_embeddings, embedding_dim)
        embedding.uniform_(-init_bound, init_bound)
        self.register_buffer("embedding", embedding)
        self.register_buffer("ema_count", torch.zeros(n_embeddings))
        self.register_buffer("ema_weight", self.embedding.clone())

    def encode(self, x):
        M, D = self.embedding.size()
        x_flat = x.detach().reshape(-1, D)

        distances = torch.addmm(torch.sum(self.embedding ** 2, dim=1) +
                                torch.sum(x_flat ** 2, dim=1, keepdim=True),
                                x_flat, self.embedding.t(),
                                alpha=-2.0, beta=1.0)

        indices = torch.argmin(distances.float(), dim=-1)
        quantized = F.embedding(indices, self.embedding)
        quantized = quantized.view_as(x)
        return quantized, indices.view(x.size(0), x.size(1))

    def forward(self, x):
        M, D = self.embedding.size()
        x_flat = x.detach().reshape(-1, D)

        distances = torch.addmm(torch.sum(self.embedding ** 2, dim=1) +
                                torch.sum(x_flat ** 2, dim=1, keepdim=True),
                                x_flat, self.embedding.t(),
                                alpha=-2.0, beta=1.0)

        indices = torch.argmin(distances.float(), dim=-1)
        encodings = F.one_hot(indices, M).float()
        quantized = F.embedding(indices, self.embedding)
        quantized = quantized.view_as(x)

        if self.training:
            self.ema_count = self.decay * self.ema_count + (1 - self.decay) * torch.sum(encodings, dim=0)

            n = torch.sum(self.ema_count)
            self.ema_count = (self.ema_count + self.epsilon) / (n + M * self.epsilon) * n

            dw = torch.matmul(encodings.t(), x_flat)
            self.ema_weight = self.decay * self.ema_weight + (1 - self.decay) * dw

            self.embedding = self.ema_weight / self.ema_count.unsqueeze(-1)

        e_latent_loss = F.mse_loss(x, quantized.detach())
        loss = self.commitment_cost * e_latent_loss

        quantized = x + (quantized - x).detach()

        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return quantized, loss, perplexity

# CPC Loss

class CPCLoss(nn.Module):
    def __init__(self, n_speakers_per_batch, n_utterances_per_speaker, n_prediction_steps, n_negatives, z_dim, c_dim):
        super(CPCLoss, self).__init__()
        self.n_speakers_per_batch = n_speakers_per_batch
        self.n_utterances_per_speaker = n_utterances_per_speaker
        self.n_prediction_steps = n_prediction_steps 
        self.n_negatives = n_negatives
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.predictors = nn.ModuleList([
            nn.Linear(c_dim, z_dim) for _ in range(n_prediction_steps)
        ])

    def forward(self, z, c):
        length = z.size(1) - self.n_prediction_steps

        z = z.reshape(
            self.n_speakers_per_batch,
            self.n_utterances_per_speaker,
            -1,
            self.z_dim
        )
        c = c[:, :-self.n_prediction_steps, :]

        losses, accuracies = list(), list()
        for k in range(1, self.n_prediction_steps+1):
            z_shift = z[:, :, k:length + k, :]

            Wc = self.predictors[k-1](c)
            Wc = Wc.view(
                self.n_speakers_per_batch,
                self.n_utterances_per_speaker,
                -1,
                self.z_dim
            )

            batch_index = torch.randint(
                0, self.n_utterances_per_speaker,
                size=(
                    self.n_utterances_per_speaker,
                    self.n_negatives
                ),
                device=z.device
            )
            batch_index = batch_index.view(
                1, self.n_utterances_per_speaker, self.n_negatives, 1
            )

            seq_index = torch.randint(
                1, length,
                size=(
                    self.n_speakers_per_batch,
                    self.n_utterances_per_speaker,
                    self.n_negatives,
                    length
                ),
                device=z.device
            )
            seq_index += torch.arange(length, device=z.device)
            seq_index = torch.remainder(seq_index, length)

            speaker_index = torch.arange(self.n_speakers_per_batch, device=z.device)
            speaker_index = speaker_index.view(-1, 1, 1, 1)

            z_negatives = z_shift[speaker_index, batch_index, seq_index, :]

            zs = torch.cat((z_shift.unsqueeze(2), z_negatives), dim=2)

            f = torch.sum(zs * Wc.unsqueeze(2) / math.sqrt(self.z_dim), dim=-1)
            f = f.view(
                self.n_speakers_per_batch * self.n_utterances_per_speaker,
                self.n_negatives + 1,
                -1
            )

            labels = torch.zeros(
                self.n_speakers_per_batch * self.n_utterances_per_speaker, length,
                dtype=torch.long, device=z.device
            )

            loss = F.cross_entropy(f, labels)

            accuracy = f.argmax(dim=1) == labels
            accuracy = torch.mean(accuracy.float())

            losses.append(loss)
            accuracies.append(accuracy.item())

        loss = torch.stack(losses).mean()
        return loss, accuracies

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

# Scheduler

class WarmupScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs, initial_lr, max_lr, milestones, gamma=0.1, last_epoch=-1):
        assert warmup_epochs < milestones[0]
        self.warmup_epochs = warmup_epochs
        self.milestones = Counter(milestones)
        self.gamma = gamma

        initial_lrs = self._format_param("initial_lr", optimizer, initial_lr)
        max_lrs = self._format_param("max_lr", optimizer, max_lr)
        if last_epoch == -1:
            for idx, group in enumerate(optimizer.param_groups):
                group["initial_lr"] = initial_lrs[idx]
                group["max_lr"] = max_lrs[idx]

        super(WarmupScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", DeprecationWarning)

        if self.last_epoch <= self.warmup_epochs:
            pct = self.last_epoch / self.warmup_epochs
            return [
                (group["max_lr"] - group["initial_lr"]) * pct + group["initial_lr"]
                for group in self.optimizer.param_groups]
        else:
            if self.last_epoch not in self.milestones:
                return [group['lr'] for group in self.optimizer.param_groups]
            return [group['lr'] * self.gamma ** self.milestones[self.last_epoch]
                    for group in self.optimizer.param_groups]

    @staticmethod
    def _format_param(name, optimizer, param):
        """Return correctly formatted lr/momentum for each param group."""
        if isinstance(param, (list, tuple)):
            if len(param) != len(optimizer.param_groups):
                raise ValueError("expected {} values for {}, got {}".format(
                    len(optimizer.param_groups), name, len(param)))
            return param
        else:
            return [param] * len(optimizer.param_groups)