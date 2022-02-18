import behavior_benchmarks.metrics as metrics
import numpy as np
import os
import yaml

def perform_evaluation(y_true, y_pred, config, output_fp = None, choices = None, probs = None):
  
  evaluation_dict = {}
  
  ## subselect to remove frames with unknown label
 
  unknown_label = config['metadata']['label_names'].index('unknown')
  mask = y_true != unknown_label
    
  ## Compute evaluation metrics
  
  # information-theoretic
  y_true_sub = y_true[mask]
  y_pred_sub = y_pred[mask]
  homogeneity = metrics.homogeneity(y_true_sub, y_pred_sub)
  evaluation_dict['homogeneity'] = float(homogeneity)
  
  # mapping-based
  num_clusters = config['num_clusters']
  num_classes = len(config['metadata']['label_names'])
  boundary_tolerance_frames = int(config['metadata']['sr'] * config['evaluation']['boundary_tolerance_sec'])
  
  mapping_based, choices, probs = metrics.mapping_based_scores(y_true, 
                                                               y_pred, 
                                                               num_clusters, 
                                                               num_classes, 
                                                               boundary_tolerance_frames = boundary_tolerance_frames, 
                                                               unknown_value = unknown_label,
                                                               choices = choices,
                                                               probs = probs
                                                              )
  evaluation_dict['averaged_scores'] = mapping_based['averaged_scores']
  evaluation_dict['MAP_scores'] = mapping_based['MAP_scores']
  
  ## Save
  
  if output_fp is not None:
    with open(output_fp, 'w') as file:
      yaml.dump(evaluation_dict, file)
      
  ## In any case, return as a dict    
  return evaluation_dict, choices, probs
    