import yaml
import numpy as np
import torch
import torch.nn as nn
from BEBE.models.supervised_nn_utils import SupervisedBehaviorModel
from torch_intermediate_layer_getter import IntermediateLayerGetter as MidGetter

class harnet(SupervisedBehaviorModel):
  def __init__(self, config):
    super(harnet, self).__init__(config)
    
    self.unknown_label = config['metadata']['label_names'].index('unknown')
    self.harnet_version = self.model_config['harnet_version']  
  
    self.model = Classifier(self.n_features,
                            self.n_classes,
                            self.device,
                            self.harnet_version,
                            self.model_config['load_pretrained_weights'],
                            self.model_config['freeze_encoder'],
                           ).to(self.device)
      
    print(self.model)

class Classifier(nn.Module):
    def __init__(self, n_features, n_classes, device, harnet_version, load_pretrained_weights, freeze_encoder):
        super(Classifier, self).__init__()
        
        harnet = torch.hub.load('OxWearables/ssl-wearables', harnet_version, class_num=5, pretrained=load_pretrained_weights).to(device)
        
        self.freeze_encoder = freeze_encoder
        if freeze_encoder:
          harnet = harnet.eval()
        
        return_layers = {
            'feature_extractor.layer1': 'layer1',
            'feature_extractor.layer2': 'layer2',
            'feature_extractor.layer3': 'layer3',
        }
        self.conv = MidGetter(harnet, return_layers=return_layers, keep_output=True)
        
        n_harnet_groups = n_features // 3
        
        self.head = nn.Linear(128, n_classes).to(device)
        
        n_channels_fusion = 128*n_harnet_groups + (n_features % 3)
        self.gru = nn.GRU(n_channels_fusion, 64, num_layers = 1, bidirectional = True, batch_first = True)
        
        
    def forward(self, x):
        # X is [batch, seq_len, channels]
        
        seq_len = x.size()[-2]
        n_channels = x.size()[-1]
        
        harnet_outputs = []
        
        with torch.set_grad_enabled(self.freeze_encoder):
        
          for channel_group in range(0, n_channels, 3):
            # group triples of channels, assumes acc channels appear first
            x_channel_group = x[:,:,channel_group:channel_group+3]


            x_channel_group = torch.transpose(x_channel_group, -1, -2)
            mids, _ = self.conv(x_channel_group)
            x_channel_group = mids['layer2']
            x_channel_group = torch.nn.functional.interpolate(x_channel_group, size = seq_len)
            x_channel_group = torch.transpose(x_channel_group, -1, -2)

            harnet_outputs.append(x_channel_group)
        
          harnet_outputs = torch.cat(harnet_outputs, dim = -1)
        
          if (n_channels % 3) > 0:
            x_extra = x[:,:,channel_group:]
            x = torch.cat([harnet_outputs, x_extra], dim = -1)
            
          else:
            x = harnet_outputs
        
        x, _ = self.gru(x)
        x = self.head(x)
        
        x = torch.transpose(x, -1, -2)
               
        return x  

