import behavior_benchmarks.visualization as bbvis
import behavior_benchmarks.evaluation.metrics as metrics
import os
import numpy as np
import pandas as pd
import tqdm
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
  label_names = config['metadata']['label_names']
  # boundary_tolerance_frames = int(config['metadata']['sr'] * config['evaluation']['boundary_tolerance_sec'])
  
  # scores for supervised model
  if config['unsupervised'] == False:
    supervised = True
  else: 
    supervised = False
  
  mapping_based, choices, probs = metrics.mapping_based_scores(y_true, 
                                                               y_pred, 
                                                               num_clusters, 
                                                               label_names, 
                                                               # boundary_tolerance_frames = boundary_tolerance_frames, 
                                                               unknown_value = unknown_label,
                                                               choices = choices,
                                                               probs = probs,
                                                               # n_samples = n_samples, 
                                                               supervised = supervised
                                                              )
  for key in mapping_based:
    evaluation_dict[key] = mapping_based[key]
  
  ## Save
  
  if output_fp is not None:
    with open(output_fp, 'w') as file:
      yaml.dump(evaluation_dict, file)
      
  ## In any case, return as a dict    
  return evaluation_dict, choices, probs

def generate_predictions(model, config):
  # Generate predictions for each file
  
  if config['unsupervised']:
    to_consider = [config['dev_file_ids'], config['test_file_ids']]
  else:
    to_consider = [config['train_file_ids'], config['val_file_ids'], config['test_file_ids']]
  
  for file_ids in to_consider:
    print("Generating predictions & latents based on trained model")
    all_predictions = []
    all_labels = []

    for filename in tqdm.tqdm(file_ids):
      fp = config['file_id_to_model_input_fp'][filename]
      predictions, latents = model.predict_from_file(fp)
      
      if config['predict_and_evaluate']:
        predictions_fp = os.path.join(config['predictions_dir'], filename)
        np.savetxt(predictions_fp, predictions.astype('int'), fmt='%3i', delimiter=",")
        # np.save(predictions_fp, predictions)

      if config['save_latents']:
        latents_fp = os.path.join(config['latents_output_dir'], filename)
        np.savetxt(latents_fp, latents, delimiter=",")
        # np.save(latents_fp, latents)

def generate_evaluations(config):
  print("saving model outputs to %s " % config['output_dir'])
  choices = None # choices and probs are parameters for the mapping based metrics, we discover them using the train set on the first loop through
  probs = None
  
  if not config['predict_and_evaluate']:
    return None
  
  if config['unsupervised']:
    to_consider = [config['dev_file_ids'], config['test_file_ids']]
  else:
    to_consider = [config['train_file_ids'], config['val_file_ids'], config['test_file_ids']]
  
  for file_ids in to_consider:
    all_predictions_dict = {}
    all_labels_dict = {}
    all_predictions = []
    all_labels = []

    for filename in file_ids:      
      predictions_fp = os.path.join(config['predictions_dir'], filename)
      if not os.path.exists(predictions_fp):
        raise ValueError("you need to save off all the model predictions before performing evaluation")
      
      predictions = np.genfromtxt(predictions_fp, delimiter = ',') #np.load(predictions_fp)
      predictions = list(predictions)

      labels_idx = config['metadata']['clip_column_names'].index('label')
      data_fp = config['file_id_to_data_fp'][filename]
      labels = list(np.genfromtxt(data_fp, delimiter = ',')[:, labels_idx]) #list(np.load(data_fp)[:, labels_idx])
      
      clip_id = filename.split('.')[0]
      individual_id = config['metadata']['clip_id_to_individual_id'][clip_id]
      
      if individual_id in all_predictions_dict:
        all_predictions_dict[individual_id].extend(predictions)
        all_labels_dict[individual_id].extend(labels)
      else:
        all_predictions_dict[individual_id] = predictions
        all_labels_dict[individual_id] = labels
      
      all_predictions.extend(predictions)
      all_labels.extend(labels)
        
    # Per-individual evaluation
    label_names = config['metadata']['label_names'].copy()
    label_names.remove('unknown')
    
    individual_f1s = {label_name : [] for label_name in label_names}
    # individual_precs = {label_name : [] for label_name in label_names}
    # individual_recs = {label_name : [] for label_name in label_names}
    
    for individual_id in all_predictions_dict:
      predictions = np.array(all_predictions_dict[individual_id])
      labels = np.array(all_labels_dict[individual_id])
      
      individual_eval_dict, _, _ = perform_evaluation(labels, predictions, config, output_fp = None, choices = None, probs = None)
      
      for label_name in label_names:
        individual_f1s[label_name].append(individual_eval_dict['MAP_scores']['MAP_classification_f1'][label_name])
        # individual_precs[label_name].append(individual_eval_dict['MAP_scores']['MAP_classification_precision'][label_name])
        # individual_recs[label_name].append(individual_eval_dict['MAP_scores']['MAP_classification_recall'][label_name])

    # Evaluate    
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    
    if file_ids == config['train_file_ids']:
      eval_output_fp = os.path.join(config['output_dir'], 'train_eval.yaml')
      confusion_target_fp = os.path.join(config['visualization_dir'], "train_confusion_matrix.png")
      f1_consistency_target_fp = os.path.join(config['visualization_dir'], "train_f1_consistency.png")
    elif file_ids == config['val_file_ids']:
      eval_output_fp = os.path.join(config['output_dir'], 'val_eval.yaml')
      confusion_target_fp = os.path.join(config['visualization_dir'], "val_confusion_matrix.png")
      f1_consistency_target_fp = os.path.join(config['visualization_dir'], "val_f1_consistency.png")
    elif file_ids == config['dev_file_ids']:
      eval_output_fp = os.path.join(config['output_dir'], 'dev_eval.yaml')
      confusion_target_fp = os.path.join(config['visualization_dir'], "dev_confusion_matrix.png")
      f1_consistency_target_fp = os.path.join(config['visualization_dir'], "dev_f1_consistency.png")
    elif file_ids == config['test_file_ids']:
      eval_output_fp = os.path.join(config['output_dir'], 'test_eval.yaml')
      confusion_target_fp = os.path.join(config['visualization_dir'], "test_confusion_matrix.png")
      f1_consistency_target_fp = os.path.join(config['visualization_dir'], "test_f1_consistency.png")
      

    if file_ids == config['dev_file_ids'] or file_ids == config['train_file_ids']:
      eval_dict, choices, probs = perform_evaluation(all_labels, all_predictions, config, output_fp = eval_output_fp, choices = choices, probs = probs)
    else: 
      eval_dict, _, _ = perform_evaluation(all_labels, all_predictions, config, output_fp = eval_output_fp, choices = choices, probs = probs)

    # Save confusion matrix
    bbvis.confusion_matrix(all_labels, all_predictions, config, target_fp = confusion_target_fp)
    
    # Save consistency plot
    bbvis.consistency_plot(individual_f1s, eval_dict['MAP_scores']['MAP_classification_f1'], config, target_fp = f1_consistency_target_fp)
  
  # Save example figures
  rng = np.random.default_rng(seed = 607)  # we want to plot segments chosen a bit randomly, but also consistently

  for file_ids in to_consider:
    for filename in list(rng.choice(file_ids, min(3, len(file_ids)), replace = False)):
      predictions_fp = os.path.join(config['predictions_dir'], filename)
      track_length = len(np.genfromtxt(predictions_fp, delimiter = ','))#len(np.load(predictions_fp))
      if file_ids == config['train_file_ids']:
        target_filename = filename.split('.')[0] + '-train-track_visualization.png'
      elif file_ids == config['val_file_ids']:
        target_filename = filename.split('.')[0] + '-val-track_visualization.png'
      elif file_ids == config['dev_file_ids']:
        target_filename = filename.split('.')[0] + '-dev-track_visualization.png'
      elif file_ids == config['test_file_ids']:
        target_filename = filename.split('.')[0] + '-test-track_visualization.png'
      target_fp = os.path.join(config['visualization_dir'], target_filename)
      data_fp = config['file_id_to_data_fp'][filename]
      if track_length <= 20000:
        bbvis.plot_track(data_fp, predictions_fp, config, eval_dict, target_fp = target_fp, start_sample = max(0, track_length - 20000), end_sample = track_length)
      else:
        start_sample = rng.integers(0, high = track_length - 20000)
        bbvis.plot_track(data_fp, predictions_fp, config, eval_dict, target_fp = target_fp, start_sample = start_sample, end_sample = start_sample + 20000)
