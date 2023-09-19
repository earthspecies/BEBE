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
        
        if load_pretrained_weights:
          harnet = torch.hub.load('OxWearables/ssl-wearables', harnet_version, class_num=5, pretrained=True).to(device)
        else:
          harnet = torch.hub.load('OxWearables/ssl-wearables', harnet_version, class_num=5, pretrained=False).to(device)
        
        self.freeze_encoder = freeze_encoder
        if freeze_encoder:
          harnet = harnet.eval()
        
        return_layers = {
            'feature_extractor.layer1': 'layer1',
            'feature_extractor.layer2': 'layer2',
            'feature_extractor.layer3': 'layer3',
        }
        self.conv = MidGetter(harnet, return_layers=return_layers, keep_output=True)
        
        self.head = nn.Linear(128, n_classes).to(device)
        
        n_channels_fusion = 128 + n_features-3
        self.gru = nn.GRU(n_channels_fusion, 64, num_layers = 1, bidirectional = True, batch_first = True)
        
        
    def forward(self, x):
        # X is [batch, seq_len, channels]
        
        seq_len = x.size()[-2]
        n_channels = x.size()[-1]
        
        if n_channels>3:
          x_extra = x[:,:,3:]
          x = x[:,:,:3]
        
        if self.freeze_encoder:
          with torch.no_grad():
            x = torch.transpose(x, -1, -2)
            mids, _ = self.conv(x)
            x = mids['layer2']
            x = torch.nn.functional.interpolate(x, size = seq_len)
            x = torch.transpose(x, -1, -2)
            if n_channels>3:
              x = torch.cat([x, x_extra], dim = -1)
        else:
          x = torch.transpose(x, -1, -2)
          mids, _ = self.conv(x)
          x = mids['layer2']
          x = torch.nn.functional.interpolate(x, size = seq_len)
          x = torch.transpose(x, -1, -2)
          if n_channels>3:
            x = torch.cat([x, x_extra], dim = -1)
          
        
        x, _ = self.gru(x)
        x = self.head(x)
        
        x = torch.transpose(x, -1, -2)
               
        return x  

