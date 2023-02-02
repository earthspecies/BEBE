import yaml
import numpy as np
import torch
import torch.nn as nn
from BEBE.models.supervised_nn_utils import SupervisedBehaviorModel

class CRNN(SupervisedBehaviorModel):
  def __init__(self, config):
    super(CRNN, self).__init__(config)
    
    self.unknown_label = config['metadata']['label_names'].index('unknown')
    self.conv_depth = self.model_config['conv_depth']
    self.ker_size = self.model_config['ker_size']
    self.dilation = self.model_config['dilation']
    self.gru_depth = self.model_config['gru_depth']
    self.gru_hidden_size = self.model_config['gru_hidden_size']
    self.hidden_size = self.model_config['hidden_size']
    
    self.model = Classifier(self.n_features,
                            self.n_classes,
                            self.conv_depth,
                            self.ker_size,
                            self.hidden_size,
                            self.dilation,
                            self.gru_depth,
                            self.gru_hidden_size,
                            dropout = self.dropout,
                            blur_scale = self.blur_scale,
                            jitter_scale = self.jitter_scale).to(self.device)
    
    print(self.model)
    print('Model parameters:')
    print(self._count_parameters())
  
class Classifier(nn.Module):
    def __init__(self, n_features, n_classes, conv_depth, ker_size, hidden_size, dilation, gru_depth, gru_hidden_size, dropout, blur_scale = 0, jitter_scale = 0):
        super(Classifier, self).__init__()
        self.blur_scale = blur_scale
        self.jitter_scale = jitter_scale
        self.bn = nn.BatchNorm1d(n_features)
        
        n_head_input_features = n_features
        n_gru_input_features = n_features
        self.conv_depth = conv_depth
        self.conv = []
        if self.conv_depth > 0:
          n_head_input_features = hidden_size
          n_gru_input_features = hidden_size
          self.conv = [_conv_block_1d(n_features, hidden_size, ker_size, dilation = dilation)]
          for i in range(conv_depth - 1):
            self.conv.append(_conv_block_1d(hidden_size, hidden_size, ker_size, dilation = dilation))
        
        self.conv = nn.ModuleList(self.conv)
  
        self.gru_depth = gru_depth      
        if self.gru_depth > 0:
          n_head_input_features = gru_hidden_size * 2
          self.gru = nn.GRU(n_gru_input_features, gru_hidden_size, num_layers = gru_depth, bidirectional = True, batch_first = True, dropout = dropout)
        
        self.head = nn.Linear(n_head_input_features, n_classes)
        
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
        
        x = torch.transpose(x, -1, -2)
        
        if self.gru_depth > 0:
          x, _ = self.gru(x)
        logits = self.head(x)
        logits = torch.transpose(logits, -1, -2)
        return logits  

def _conv_block_1d(in_channels, out_channels, kernel_size, dilation = 1):
  block = nn.Sequential(
    nn.Conv1d(in_channels, out_channels, kernel_size, dilation = dilation, padding='same', bias=False),
    torch.nn.BatchNorm1d(out_channels),
    nn.ReLU()
  )
  return block
