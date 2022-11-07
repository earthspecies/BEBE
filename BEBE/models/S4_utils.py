import torch
import torch.nn as nn
from BEBE.applications.S4.S4 import S4

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
    def __init__(self, n_features, hidden_size, state_size, n_s4_blocks, downsample_rate, feature_expansion_factor, dropout, blur_scale = 0, jitter_scale = 0):
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
        self.output_dims = (feature_expansion_factor ** 2) * hidden_size
        
        #self.head = nn.Conv1d((feature_expansion_factor ** 2) * hidden_size, n_clusters, 1, padding = 'same')
        
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
        #logits = self.head(x) # -> [batch, n_clusters, seq_len]
        
        x = nn.functional.interpolate(x, size=seq_len, mode='nearest-exact')
        latents = torch.transpose(x, 1,2) # -> [batch, seq_len, n_features]
        return latents