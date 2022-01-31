import behavior_benchmarks.metrics as metrics
import numpy as np
import os
import yaml

def perform_evaluation(y_true, y_pred, config, output_fp = None):
  
  evaluation_dict = {}
  
  ## subselect to remove frames with unknown label
 
  unknown_label = config['metadata']['label_names'].index('unknown')
  mask = y_true != unknown_label
  
  y_true_sub = y_true[mask]
  y_pred_sub = y_pred[mask]
  
  ## Compute evaluation metrics
  
  uncertainty = metrics.Thiel_U(y_true_sub, y_pred_sub)
  evaluation_dict['uncertainty'] = float(uncertainty)
  
  ## Save
  
  if output_fp == None:
    print(evaluation_dict)
    
  else:
    with open(output_fp, 'w') as file:
      yaml.dump(evaluation_dict, file)
    
  return evaluation_dict
    