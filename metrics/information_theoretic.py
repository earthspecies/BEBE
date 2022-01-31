import scipy
import scipy.io as sio
import glob
import pandas as pd
import numpy as np
from pathlib import Path
import os
from sklearn.metrics import mutual_info_score

## various information theoretic metrics

def _shannon_entropy(seq):
    N = len(seq)
    entropy = 0
    for i in np.unique(seq):
        p = np.sum(seq == i) / N
        entropy += -p*np.log(p)
    return entropy
  
def Thiel_U(labels_coarse, labels_fine):
    return mutual_info_score(labels_coarse, labels_fine)/ _shannon_entropy(labels_coarse)
