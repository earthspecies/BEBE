import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import glob
import yaml

class dataloader:
  def __init__(self, data_dir):
    self.data_dir = data_dir
    
    metadata_fp = os.path.join(data_dir, 'dataset_metadata.yaml')
    with open(metadata_fp) as file:
      self.metadata = yaml.load(file, Loader=yaml.FullLoader)
  
  def load_clip(self, clip_id):
    clip_fp = os.path.join(self.data_dir, 'clip_data', clip_id + '.npy')
    clip = np.load(clip_fp)
    return clip