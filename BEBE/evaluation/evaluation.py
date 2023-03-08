import BEBE.visualization as visualization
import BEBE.evaluation.metrics as metrics
import os
import numpy as np
import pandas as pd
import tqdm
import yaml
import warnings
from sklearn.metrics import homogeneity_score

def perform_evaluation(y_true, y_pred, metadata, num_clusters, unsupervised, output_fp = None, mapping_dict = None, target_time_scale_sec = 1.):
  # y_true, y_pred (list): list of integers, corresponding to class/cluster index
  # mapping_dict (dict): sends cluster indices (int) to behavior label indices (int)
  
  evaluation_dict = {}

  ## subselect to remove frames with unknown label
  unknown_label = metadata['label_names'].index('unknown')
  sr = metadata['sr']
  mask = (y_true != unknown_label)
    
  ## Compute evaluation metrics
  # information-theoretic
  y_true_sub = y_true[mask]
  y_pred_sub = y_pred[mask]
  homogeneity = homogeneity_score(y_true_sub, y_pred_sub)
  evaluation_dict['homogeneity'] = float(homogeneity)
  
  # mapping-based
  num_clusters = num_clusters
  label_names = metadata['label_names']
  
  scores, mapping_dict = metrics.mapping_based_scores(y_true,
                                                      y_pred, 
                                                      num_clusters, 
                                                      label_names, 
                                                      unknown_value = unknown_label,
                                                      mapping_dict = mapping_dict,
                                                      supervised = not unsupervised,
                                                      target_time_scale_sec = target_time_scale_sec,
                                                      sr = sr
                                                     )
  for key in scores:
    evaluation_dict[key] = scores[key]
  
  ## Save
  if output_fp is not None:
    with open(output_fp, 'w') as file:
      yaml.dump(evaluation_dict, file)
      
  ## Return as a dict    
  return evaluation_dict, mapping_dict

def generate_predictions(model, config):
  # Generate predictions for each file
  
  to_consider = [config['train_data_fp'], config['val_data_fp'], config['test_data_fp']]
  
  for data_fp in to_consider:
    print("Generating predictions & based on trained model")
    all_predictions = []
    all_labels = []

    for filename in tqdm.tqdm(data_fp):
      predictions, latents = model.predict_from_file(filename)

      predictions_fp = os.path.join(config['predictions_dir'], filename.split('/')[-1])
      pd.DataFrame(predictions.astype('int')).to_csv(predictions_fp, index = False, header = False)
      
def generate_evaluations_standalone(metadata,
                                    output_dir, 
                                    visualization_dir,
                                    unsupervised, 
                                    num_clusters,
                                    train_data_fp,
                                    val_data_fp,
                                    test_data_fp,
                                    predictions_dir,
                                    dataset_dir):
  # metadata (dict): dataset metadata
  # output_dir (str): path to directory where you want to save evaluation files
  # visualization_dir (str): path to directory where you want to save visualizations
  # unsupervised (bool): if model is unsupervised
  # num_clusters (int): number of clusters / classes output by model
  # train/val/test_data_fp (each a list of strs): filenames for the different data splits.
  # predictions_dir (str): path to directory where predictions are stored
  # dataset_dir (str): path to dataset directory
  
  print(f"saving model outputs to {output_dir}")
  
  
  if unsupervised and num_clusters != max(20, 4*(len(metadata['label_names'])-1)):
    warnings.warn("Using a non-default number of clusters N. Results using different values of N should not be compared to each other.")
  
  to_consider = [train_data_fp, val_data_fp, test_data_fp]
  
  # mapping_dict is generated through contingency analysis of train data  
  mapping_dict = None 
  
  for data_fp in to_consider:
    if len(data_fp) == 0:
      continue
    all_predictions_dict = {}
    all_labels_dict = {}
    all_predictions = []
    all_labels = []
    
    #######
    # Load predictions
    #######
    
    for fp in data_fp:
      filename = fp.split('/')[-1]
      predictions_fp = os.path.join(predictions_dir, filename)
      if not os.path.exists(predictions_fp):
        raise ValueError("you need to save off all the model predictions before performing evaluation")
      
      predictions = pd.read_csv(predictions_fp, delimiter = ',', header = None).values.flatten()
      predictions = list(predictions)

      labels_idx = metadata['clip_column_names'].index('label')
      labels = list(pd.read_csv(fp, delimiter = ',', header = None).values[:, labels_idx].flatten())
      clip_id = filename.split('/')[-1].split('.')[0]
      individual_id = metadata['clip_id_to_individual_id'][clip_id]
      
      if individual_id in all_predictions_dict:
        all_predictions_dict[individual_id].extend(predictions)
        all_labels_dict[individual_id].extend(labels)
      else:
        all_predictions_dict[individual_id] = predictions
        all_labels_dict[individual_id] = labels
      
      all_predictions.extend(predictions)
      all_labels.extend(labels)
    
    ######
    # Perform Evaluation
    ######
    
    eval_dict = {}
    
    # Overall evaluation: Lump all individuals together
    
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    time_scale = metadata['mean_overall_dur_sec']
    label_names = metadata['label_names'].copy()
    label_names.remove('unknown')
      
    if data_fp == train_data_fp:
      eval_dict['overall_scores'], mapping_dict = perform_evaluation(all_labels, all_predictions, metadata, num_clusters, unsupervised,  mapping_dict = mapping_dict, target_time_scale_sec = time_scale)
    else: 
      eval_dict['overall_scores'], _ = perform_evaluation(all_labels, all_predictions, metadata, num_clusters, unsupervised, mapping_dict = mapping_dict, target_time_scale_sec = time_scale)
      
    # Per-individual evaluation: Treat individuals as separate test sets.
    # This gives us more test replicates, to get a better sense of model variance across different individuals
    ## We also compute 'individualized' f1 scores, which uses a different method of assigning clusters to labels
    ## Note individualized scores are only relevant to unsupervised models.
    
    individual_f1s_individualized = {}
    individual_f1s_individualized['per_label'] = {label_name : [] for label_name in label_names} # scores for individualized cluster -> label assignment
    individual_scores = {'macro_f1s' : [], 'macro_precisions' : [], 'macro_recalls' : [], 'time_scale_ratios' : []} # scores for each individual, all using the same cluster -> label assignment we found earlier
    macro_f1s_individualized = [] # scores for each individual, all using the same cluster -> label assignment we found earlier

    for individual_id in sorted(all_predictions_dict.keys()):
      predictions = np.array(all_predictions_dict[individual_id])
      labels = np.array(all_labels_dict[individual_id])
      individual_time_scale = metadata['mean_dur_sec_by_individual'][individual_id]

      individual_eval_dict_individualized, _ = perform_evaluation(labels, predictions, metadata, num_clusters, unsupervised, output_fp = None, mapping_dict = None, target_time_scale_sec = individual_time_scale)

      for label_name in label_names:
          individual_f1s_individualized['per_label'][label_name].append(individual_eval_dict_individualized['classification_f1'][label_name])
          
      individual_eval_dict, _ = perform_evaluation(labels, predictions, metadata, num_clusters, unsupervised, output_fp = None, mapping_dict = mapping_dict, target_time_scale_sec = individual_time_scale)

      individual_scores['macro_f1s'].append(individual_eval_dict['classification_f1_macro'])
      individual_scores['macro_precisions'].append(individual_eval_dict['classification_precision_macro'])
      individual_scores['macro_recalls'].append(individual_eval_dict['classification_recall_macro'])
      individual_scores['time_scale_ratios'].append(individual_eval_dict['time_scale_ratio'])
      macro_f1s_individualized.append(individual_eval_dict_individualized['classification_f1_macro'])
                                           
    eval_dict['individual_scores'] = individual_scores
    
    ######
    ## Save off evaluations
    ######
    
    if data_fp == train_data_fp:
      eval_output_fp = os.path.join(output_dir, 'train_eval.yaml')
      confusion_target_fp = os.path.join(visualization_dir, "train_confusion_matrix.png")
      f1_consistency_target_fp = os.path.join(visualization_dir, "train_f1_consistency.png")
      f1_consistency_numerical_target_fp = os.path.join(output_dir, "train_f1_consistency.yaml")
    elif data_fp == val_data_fp:
      eval_output_fp = os.path.join(output_dir, 'val_eval.yaml')
      confusion_target_fp = os.path.join(visualization_dir, "val_confusion_matrix.png")
      f1_consistency_target_fp = os.path.join(visualization_dir, "val_f1_consistency.png")
      f1_consistency_numerical_target_fp = os.path.join(output_dir, "val_f1_consistency.yaml")
    elif data_fp == test_data_fp:
      eval_output_fp = os.path.join(output_dir, 'test_eval.yaml')
      confusion_target_fp = os.path.join(visualization_dir, "test_confusion_matrix.png")
      f1_consistency_target_fp = os.path.join(visualization_dir, "test_f1_consistency.png")
      f1_consistency_numerical_target_fp = os.path.join(output_dir, "test_f1_consistency.yaml")
    
      
    with open(eval_output_fp, 'w') as file:
      yaml.dump(eval_dict, file)
      
    # Save confusion matrix
    visualization.confusion_matrix(all_labels, all_predictions, metadata, num_clusters, unsupervised, target_fp = confusion_target_fp)
    
    # Save 'individualized' scores, i.e. compare individual vs overall performance for different ways of assigning clusters to labels
    overall_f1_eval_scores = eval_dict['overall_scores']['classification_f1']
      
    visualization.consistency_plot(individual_f1s_individualized['per_label'], overall_f1_eval_scores, target_fp = f1_consistency_target_fp)
    
    classes = list(individual_f1s_individualized['per_label'].keys()).copy()
    individual_f1s_individualized['mean_f1_individualized'] = {k: float(np.mean(individual_f1s_individualized['per_label'][k])) for k in classes} # first average across individuals
    individual_f1s_individualized['macro_f1s_individualized'] = macro_f1s_individualized
    with open(f1_consistency_numerical_target_fp, 'w') as file:
      yaml.dump(individual_f1s_individualized, file)
  # Save example figures
  rng = np.random.default_rng(seed = 607)  # plot segments chosen randomly, but consistently across trials

  for data_fp in to_consider:
    if len(data_fp) == 0:
      continue
    for i, fp in enumerate(rng.choice(data_fp, 3, replace = True)):
      filename = fp.split('/')[-1]
      predictions_fp = os.path.join(predictions_dir, filename)
      track_length = len(pd.read_csv(predictions_fp, delimiter = ',', header = None))
      if data_fp == train_data_fp:
        target_filename = filename.split('.')[0] + '-' + str(i) + '-train-track_visualization.png'
      elif data_fp == val_data_fp:
        target_filename = filename.split('.')[0] + '-' + str(i) + '-val-track_visualization.png'
      elif data_fp == test_data_fp:
        target_filename = filename.split('.')[0] + '-' + str(i) + '-test-track_visualization.png'
      target_fp = os.path.join(visualization_dir, target_filename)
      data_fp = os.path.join(dataset_dir, 'clip_data', filename)
      if track_length <= 20000:
        visualization.plot_track(data_fp, predictions_fp, metadata, num_clusters, unsupervised, eval_dict, target_fp = target_fp, start_sample = max(0, track_length - 20000), end_sample = track_length)
      else:
        start_sample = rng.integers(0, high = track_length - 20000)
        visualization.plot_track(data_fp, predictions_fp, metadata, num_clusters, unsupervised, eval_dict, target_fp = target_fp, start_sample = start_sample, end_sample = start_sample + 20000)

def generate_evaluations(config):
  # Generates numerical metrics as well as visualizations
  # Assumes config has been expanded by expand_config in experiment_setup.py
  output_dir = config['output_dir']
  unsupervised = config['unsupervised']
  num_clusters = config['num_clusters']
  metadata = config['metadata']
  predictions_dir = config['predictions_dir']
  dataset_dir = config['dataset_dir']
  train_data_fp = config['train_data_fp']
  val_data_fp = config['val_data_fp']
  test_data_fp = config['test_data_fp']
  visualization_dir = config['visualization_dir']
  
  generate_evaluations_standalone(metadata,
                                  output_dir, 
                                  visualization_dir,
                                  unsupervised, 
                                  num_clusters,
                                  train_data_fp,
                                  val_data_fp,
                                  test_data_fp,
                                  predictions_dir,
                                  dataset_dir)  
