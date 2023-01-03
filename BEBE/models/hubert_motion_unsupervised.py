import yaml
import numpy as np
import torch
import torch.nn as nn
from BEBE.models.model_superclass import BehaviorModel
from sklearn.cluster import KMeans
import pandas as pd
import fairseq
import torch.nn.functional as F


class hubert_motion_unsupervised(BehaviorModel):
  def __init__(self, config):
    super(hubert_motion_unsupervised, self).__init__(config)
    
    self.unknown_label = config['metadata']['label_names'].index('unknown')
    
    ## Todo add blur etc 
    
    self.model_path = self.model_config['model_path']
    assert self.model_path is not None, 'Need to specify path to pretrained model checkpoint'
    
    
    # Get cpu or gpu device for training.
    self.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {self.device} device")
    
    self.fe = HubertMotionUnsupervised(self.model_config['model_path']).to(self.device)
    self.model = KMeans(n_clusters = self.config['num_clusters'], verbose = 0, max_iter = self.model_config['max_iter'], n_init = self.model_config['n_init'])
    
    print(self.model)
    
  def load_model_inputs(self, filepath, read_latents = False, downsample = 1):
    x =  pd.read_csv(filepath, delimiter = ',', header = None).values[:, self.cols_included]
    with torch.no_grad():
      x = torch.from_numpy(x).float().cuda()
      L = x.size(0)
      if False:
          x = torch.transpose(x, -1, -2)
          x = F.layer_norm(x, (x.size(dim = -1),))
          x = torch.transpose(x, -1, -2)
      x = torch.unsqueeze(x, 0)
      chunk_size = 10000
      feat = []
      for start in range(0, x.size(1)-chunk_size, chunk_size):
          if start + 2*chunk_size > x.size(1):
            cur_chunk_size = 2*chunk_size
          else:
            cur_chunk_size = chunk_size
          x_chunk = x[:, start : start + cur_chunk_size, :]
          feat_chunk = self.fe(x_chunk)
          feat.append(feat_chunk)
    all_feat = torch.cat(feat, 2).squeeze(0).T
    assert all_feat.size(0) == L
    return all_feat[::downsample, :].cpu().numpy()
    
  def fit(self):
    ## get data. assume stored in memory for now
    if self.read_latents:
      raise NotImplementedError
    else:
      dev_fps = self.config['dev_data_fp']
    
    dev_data = [self.load_model_inputs(fp, read_latents = self.read_latents, downsample = 25) for fp in dev_fps]
    dev_data = np.concatenate(dev_data, axis = 0)
    
    print("fitting kmeans")
    self.model.fit(dev_data)
    
  def predict(self, data):
    predictions = self.model.predict(data)
    return predictions, None
    
  def save(self):
    pass # not implemented for hubert_motion


class HubertMotionUnsupervised(nn.Module):
    def __init__(self, model_path):
        super().__init__()

        models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([model_path])
        self.model = models[0]

    def forward(self, x):
        L = x.size(1)
        # print(x.size())
        out, _ = self.model.extract_features(
                source=x,
                padding_mask=None,
                mask=False,
                output_layer=12,
            )
        out = torch.transpose(out, -1, -2)
        # print(out.size())
        out = nn.functional.interpolate(out, size=L, mode='nearest-exact')
        # print(out.size())
        return out
