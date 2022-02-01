from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix as cm

def confusion_matrix(all_labels, all_predictions, config, target_fp = None):
  metadata = config['metadata']
  label_names = metadata['label_names']
  unknown_idx = label_names.index('unknown')
  
  M = cm(all_labels, all_predictions)
  to_plot_idx = np.arange(len(label_names))
  to_plot_idx = to_plot_idx[to_plot_idx != unknown_idx]
  M = M[to_plot_idx,:] # drop the unknown labels

  figure = plt.figure(figsize = (10, len(label_names) + 1))
  axes = figure.add_subplot(111)

  Mplot = axes.matshow(M)
  plt.yticks(np.arange(len(label_names)-1), [label_names[x] for x in to_plot_idx], rotation=20)
  plt.ylabel('Ground Truth Labels')
  plt.title(config['experiment_name'])
  figure.colorbar(Mplot, orientation = 'horizontal')
  
  if target_fp == None:
    plt.show()
    
  else:
    plt.savefig(target_fp); plt.close()