import BEBE.visualization as bbvis
import BEBE.evaluation.metrics as metrics
import os
import numpy as np
import pandas as pd
import tqdm
import yaml

def perform_evaluation(y_true, y_pred, config, output_fp = None, choices = None, probs = None):
  # y_true, y_pred: list of integers
  # choices, probs: dictionaries discovered by comparing train predictions with ground truth labels  
  
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
  
  # scores for supervised model
  if config['unsupervised'] == False:
    supervised = True
  else: 
    supervised = False
  
  scores, choices, probs = metrics.mapping_based_scores(y_true,
                                                        y_pred, 
                                                        num_clusters, 
                                                        label_names, 
                                                        unknown_value = unknown_label,
                                                        choices = choices,
                                                        probs = probs,
                                                        supervised = supervised
                                                       )
  for key in scores:
    evaluation_dict[key] = scores[key]
  
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

      predictions_fp = os.path.join(config['predictions_dir'], filename)
      np.savetxt(predictions_fp, predictions.astype('int'), fmt='%3i', delimiter=",")

      if config['save_latents']:
        latents_fp = os.path.join(config['latents_output_dir'], filename)
        np.savetxt(latents_fp, latents, delimiter=",")

def generate_evaluations(config):
  # Generates numerical metrics as well as visualizations
  # Assumes config has been expanded by expand_config in experiment_setup.py
  
  print("saving model outputs to %s " % config['output_dir'])
  
  # choices and probs are parameters for the mapping based metrics used for unsupervised models
  # we discover them using the train set on the first loop through
  
  choices = None 
  probs = None
  
  if config['unsupervised']:
    to_consider = [config['dev_file_ids'], config['test_file_ids']]
  else:
    to_consider = [config['train_file_ids'], config['val_file_ids'], config['test_file_ids']]
  
  for file_ids in to_consider:
    all_predictions_dict = {}
    all_labels_dict = {}
    all_predictions = []
    all_labels = []
    
    # Load predictions
    
    for filename in file_ids:      
      predictions_fp = os.path.join(config['predictions_dir'], filename)
      if not os.path.exists(predictions_fp):
        raise ValueError("you need to save off all the model predictions before performing evaluation")
      
      predictions = pd.read_csv(predictions_fp, delimiter = ',', header = None).values.flatten()
      predictions = list(predictions)

      labels_idx = config['metadata']['clip_column_names'].index('label')
      data_fp = config['file_id_to_data_fp'][filename]
      labels = list(pd.read_csv(data_fp, delimiter = ',', header = None).values[:, labels_idx].flatten())      
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
        
    # Overall evaluation: Lump all individuals together
    
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    
    if file_ids == config['train_file_ids']:
      eval_output_fp = os.path.join(config['output_dir'], 'train_eval.yaml')
      confusion_target_fp = os.path.join(config['visualization_dir'], "train_confusion_matrix.png")
      f1_consistency_target_fp = os.path.join(config['visualization_dir'], "train_f1_consistency.png")
      f1_consistency_numerical_target_fp = os.path.join(config['output_dir'], "train_f1_consistency.yaml")
    elif file_ids == config['val_file_ids']:
      eval_output_fp = os.path.join(config['output_dir'], 'val_eval.yaml')
      confusion_target_fp = os.path.join(config['visualization_dir'], "val_confusion_matrix.png")
      f1_consistency_target_fp = os.path.join(config['visualization_dir'], "val_f1_consistency.png")
      f1_consistency_numerical_target_fp = os.path.join(config['output_dir'], "val_f1_consistency.yaml")
    elif file_ids == config['dev_file_ids']:
      eval_output_fp = os.path.join(config['output_dir'], 'dev_eval.yaml')
      confusion_target_fp = os.path.join(config['visualization_dir'], "dev_confusion_matrix.png")
      f1_consistency_target_fp = os.path.join(config['visualization_dir'], "dev_f1_consistency.png")
      f1_consistency_numerical_target_fp = os.path.join(config['output_dir'], "dev_f1_consistency.yaml")
    elif file_ids == config['test_file_ids']:
      eval_output_fp = os.path.join(config['output_dir'], 'test_eval.yaml')
      confusion_target_fp = os.path.join(config['visualization_dir'], "test_confusion_matrix.png")
      f1_consistency_target_fp = os.path.join(config['visualization_dir'], "test_f1_consistency.png")
      f1_consistency_numerical_target_fp = os.path.join(config['output_dir'], "test_f1_consistency.yaml")
      
    if file_ids == config['dev_file_ids'] or file_ids == config['train_file_ids']:
      eval_dict, choices, probs = perform_evaluation(all_labels, all_predictions, config, choices = choices, probs = probs)
    else: 
      eval_dict, _, _ = perform_evaluation(all_labels, all_predictions, config, choices = choices, probs = probs)
      
    # Per-individual evaluation: Treat individuals as separate test sets.
    # This gives us more test replicates, to get a better sense of model variance across different individuals
    
    ## We also compute 'individualized' f1 scores, which uses a different method of assigning clusters to labels
    ## Individualized scores are only relevant to unsupervised models.
    
    label_names = config['metadata']['label_names'].copy()
    label_names.remove('unknown')
    
    individual_f1s_individualized = {}
    individual_f1s_individualized['per_label'] = {label_name : [] for label_name in label_names} # scores for individualized cluster -> label assignment
    individual_scores = {'macro_f1s' : [], 'macro_precisions' : [], 'macro_recalls' : []} # scores for each individual, all using the same cluster -> label assignment we found earlier
    macro_f1s_individualized = [] # scores for each individual, all using the same cluster -> label assignment we found earlier

    for individual_id in sorted(all_predictions_dict.keys()):
      predictions = np.array(all_predictions_dict[individual_id])
      labels = np.array(all_labels_dict[individual_id])

      individual_eval_dict_individualized, _, _ = perform_evaluation(labels, predictions, config, output_fp = None, choices = None, probs = None)

      for label_name in label_names:
        if config['unsupervised']:
          individual_f1s_individualized['per_label'][label_name].append(individual_eval_dict_individualized['MAP_scores']['MAP_classification_f1'][label_name])
        else: 
          individual_f1s_individualized['per_label'][label_name].append(individual_eval_dict_individualized['supervised_scores']['classification_f1'][label_name])
          
      individual_eval_dict, _, _ = perform_evaluation(labels, predictions, config, output_fp = None, choices = choices, probs = probs)
      if config['unsupervised']:
        individual_scores['macro_f1s'].append(individual_eval_dict['MAP_scores']['MAP_classification_f1_macro'])
        individual_scores['macro_precisions'].append(individual_eval_dict['MAP_scores']['MAP_classification_precision_macro'])
        individual_scores['macro_recalls'].append(individual_eval_dict['MAP_scores']['MAP_classification_recall_macro'])
        macro_f1s_individualized.append(individual_eval_dict_individualized['MAP_scores']['MAP_classification_f1_macro'])
      else:
        individual_scores['macro_f1s'].append(individual_eval_dict['supervised_scores']['classification_f1_macro'])
        individual_scores['macro_precisions'].append(individual_eval_dict['supervised_scores']['classification_precision_macro'])
        individual_scores['macro_recalls'].append(individual_eval_dict['supervised_scores']['classification_recall_macro'])
        macro_f1s_individualized.append(individual_eval_dict_individualized['supervised_scores']['classification_f1_macro'])
                                            
    eval_dict['individual_scores'] = individual_scores
      
    ## Save off evaluation  
      
    with open(eval_output_fp, 'w') as file:
      yaml.dump(eval_dict, file)
      
    # Save confusion matrix
    bbvis.confusion_matrix(all_labels, all_predictions, config, target_fp = confusion_target_fp)
    
    # Save 'individualized' scores, i.e. compare individual vs overall performance for different ways of assigning clusters to labels
    if config['unsupervised']:
      overall_f1_eval_scores = eval_dict['MAP_scores']['MAP_classification_f1']
    else:
      overall_f1_eval_scores = eval_dict['supervised_scores']['classification_f1']
      
    bbvis.consistency_plot(individual_f1s_individualized['per_label'], overall_f1_eval_scores, config, target_fp = f1_consistency_target_fp)
    
    classes = list(individual_f1s_individualized['per_label'].keys()).copy()
    individual_f1s_individualized['mean_f1_individualized'] = {k: float(np.mean(individual_f1s_individualized['per_label'][k])) for k in classes} # first average across individuals
    individual_f1s_individualized['macro_f1s_individualized'] = macro_f1s_individualized
    with open(f1_consistency_numerical_target_fp, 'w') as file:
      yaml.dump(individual_f1s_individualized, file)
    
  
  # Save example figures
  rng = np.random.default_rng(seed = 607)  # we want to plot segments chosen a bit randomly, but also consistently

  for file_ids in to_consider:
    for i, filename in enumerate(rng.choice(file_ids, 3, replace = True)):
      predictions_fp = os.path.join(config['predictions_dir'], filename)
      track_length = len(pd.read_csv(predictions_fp, delimiter = ',', header = None))
      if file_ids == config['train_file_ids']:
        target_filename = filename.split('.')[0] + '-' + str(i) + '-train-track_visualization.png'
      elif file_ids == config['val_file_ids']:
        target_filename = filename.split('.')[0] + '-' + str(i) + '-val-track_visualization.png'
      elif file_ids == config['dev_file_ids']:
        target_filename = filename.split('.')[0] + '-' + str(i) + '-dev-track_visualization.png'
      elif file_ids == config['test_file_ids']:
        target_filename = filename.split('.')[0] + '-' + str(i) + '-test-track_visualization.png'
      target_fp = os.path.join(config['visualization_dir'], target_filename)
      data_fp = config['file_id_to_data_fp'][filename]
      if track_length <= 20000:
        bbvis.plot_track(data_fp, predictions_fp, config, eval_dict, target_fp = target_fp, start_sample = max(0, track_length - 20000), end_sample = track_length)
      else:
        start_sample = rng.integers(0, high = track_length - 20000)
        bbvis.plot_track(data_fp, predictions_fp, config, eval_dict, target_fp = target_fp, start_sample = start_sample, end_sample = start_sample + 20000)
