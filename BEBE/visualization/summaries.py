from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix as cm

def confusion_matrix(all_labels, all_predictions, metadata, num_clusters, unsupervised, target_fp = None):
  # confusion matrix for single experiment. If unsupervised, plots the contingency matrix of clusters against ground truth labels
  label_names = metadata['label_names']
  unknown_idx = label_names.index('unknown')
  
  M = cm(all_labels, all_predictions, labels = np.arange(num_clusters), normalize = 'pred')
  to_plot_idx = np.arange(len(label_names))
  to_plot_idx = to_plot_idx[to_plot_idx != unknown_idx]
  if unsupervised:
    to_plot_start = 0
  else:
    to_plot_start = 1
    
  M = M[to_plot_idx,to_plot_start:] # drop the unknown labels

  figure = plt.figure(figsize = (10, len(label_names) + 1))
  axes = figure.add_subplot(111)

  Mplot = axes.matshow(M)
  plt.yticks(np.arange(len(label_names)-1), [label_names[x] for x in to_plot_idx], rotation=20)
  plt.ylabel('Ground Truth Labels')
  figure.colorbar(Mplot, orientation = 'horizontal')
  
  plt.tight_layout()
  
  if target_fp == None:
    plt.show()
    
  else:
    plt.savefig(target_fp); plt.close()
    
def consistency_plot(per_class_per_individual_f1s, per_class_f1s, target_fp = None):
  # per_class_per_individual_f1s (dict) : keys are behavior classes, values are lists of f1 scores, 1 per individual. 
  # per_class_f1s (dict) : keys are behavior classes, values are mean f1 scores
  
  fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9, 9))

  for i, key in enumerate(sorted(per_class_per_individual_f1s.keys())[:1]):
    xloc = [i for x in per_class_per_individual_f1s[key]]
    ax.scatter(xloc, per_class_per_individual_f1s[key], c = 'blue', label = "individual interpretation")
    ax.scatter(i, per_class_f1s[key], c = 'green', marker = "_", s = 300, label = "all-data interpretation")

  for i, key in enumerate(sorted(per_class_per_individual_f1s.keys())):
    xloc = [i for x in per_class_per_individual_f1s[key]]
    ax.scatter(xloc, per_class_per_individual_f1s[key], c = 'blue')
    ax.scatter(i, per_class_f1s[key], c = 'green', marker = "_", s = 300)

  ax.set_ylabel("f1 score")
  ax.set_xticks(np.arange(len(sorted(per_class_per_individual_f1s.keys()))), )
  ax.set_xticklabels(sorted(per_class_per_individual_f1s.keys()), rotation=45)
  ax.set_title("f1 for different methods of assigning behavior labels to clusters")
  ax.legend()
  
  plt.tight_layout()

  if target_fp == None:
    plt.show()

  else:
    plt.savefig(target_fp); plt.close()

