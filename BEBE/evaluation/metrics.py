import numpy as np
from sklearn.metrics import confusion_matrix as cm
from sklearn.metrics import precision_score, recall_score, f1_score
from matplotlib import pyplot as plt
import tqdm
import functools
import operator
import concurrent.futures
import itertools
import os
from sklearn.metrics import homogeneity_score
 
def homogeneity(labels_coarse, labels_fine):
  # alternative name for sklearn homogeneity score
  return homogeneity_score(labels_coarse, labels_fine)

def find_unknown_mask(array, unknown_value = 0, tolerance_frames = 0):
    # array: 1-dim array
    # returns: 1-dim boolean mask of array with 0's anywhere that is within tolerance_frames frames of a frame containing unknown_value
    
    array = np.array(array)
    shifted_masks = []
    shifted_masks.append(array != unknown_value)
    for shift in range(1, tolerance_frames + 1):
        right_shifted_mask = array[shift:] != unknown_value
        right_shifted_mask = np.append(right_shifted_mask, np.full(shift, False))
        left_shifted_mask = array[:-shift] != unknown_value
        left_shifted_mask = np.append(np.full(shift, False), left_shifted_mask)
        shifted_masks.append(right_shifted_mask)
        shifted_masks.append(left_shifted_mask)
    mask = functools.reduce(operator.mul, shifted_masks, 1)
    return mask.astype(bool)
  
# Discover probabilities of mapping cluster -> label based on confusion matrix
def discover_probabilities(gt, pred, num_clusters, num_classes, unknown_value = 0):
    # gt: 1-dim array of gt labels (per frame)
    # pred: 1-dim array of predicted clusters (per frame)
    # num_clusters: number of clusters allowed, we assume the clusters are numbered 0,1,2,...   
    # num_classes: number of classes allowed, includes unknown, assume numbered 0,1,2,...
    # Returns:
    # choices: dict where choices[i] is a list of allowed values to map cluster i to
    # probs: dict where probs[i] is a list of probabilites associated with the choies
    mask = find_unknown_mask(gt, unknown_value = unknown_value)
    gt_sub = gt[mask]
    pred_sub = pred[mask]
    choices = {}
    probs = {}
    for i in range(num_clusters):
        # look at frames for a given cluster
        gt_sub_i = gt_sub[pred_sub == i]
        choices[i], counts = np.unique(gt_sub_i, return_counts = True)
        if len(choices[i]) == 0:
            choices[i] = np.array([unknown_value])
            probs[i] = np.array([1.])
        else:
            probs[i] = counts / np.sum(counts)
    return choices, probs
  
# Find the best way of mapping cluster -> label
def produce_MAP_cluster_to_label(choices, probs):
    # choices: dict where choices[i] is a list of allowed values to map cluster i to
    # probs: dict where probs[i] is a list of probabilites associated with the choies
    mapping_dict = {}
    for i in choices:
        best_idx = np.argmax(probs[i])
        mapping_dict[int(i)] = int(choices[i][best_idx]) # cast as int for saving off results
    return lambda i : mapping_dict[i], mapping_dict
  
def get_MAP_scores(gt, pred, choices, probs, label_names, unknown_value=0):
    # gt: 1-dim array of gt labels (per frame)
    # pred: 1-dim array of predicted clusters (per frame)
    # choices: dict where choices[i] is a list of allowed values to map cluster i to
    # probs: dict where probs[i] is a list of probabilites associated with the choies
    # label_names: list of behavior label names
  
    # For each cluster, looks for the label with highest overlap with that cluster
    # Maps that cluster to that label, and computes precision, recall, etc
    # Returns: dict including the MAP mapping from clusters to labels
    results = {}
    pred_list = list(pred)
    mapping, mapping_dict = produce_MAP_cluster_to_label(choices, probs)
    results['MAP_mapping_dict'] = mapping_dict    
    pred_mapped = np.array(list(map(mapping, pred_list)))
    
    ### Get optimized classification scores
    mask = find_unknown_mask(gt, unknown_value = unknown_value)
    gt_sub = gt[mask]
    pred_sub = pred_mapped[mask]
    
    labels = np.arange(len(label_names))
    labels = labels[labels != unknown_value]
    
    precisions = precision_score(gt_sub, pred_sub, labels = labels, average = None, zero_division =1)
    results['MAP_classification_precision'] = {label_names[labels[i]] : float(precisions[i]) for i in range(len(precisions))}
    results['MAP_classification_precision_macro'] = float(np.mean(precisions))
    
    recalls = recall_score(gt_sub, pred_sub, labels = labels, average = None, zero_division =1)
    results['MAP_classification_recall'] = {label_names[labels[i]] : float(recalls[i]) for i in range(len(recalls))}
    results['MAP_classification_recall_macro'] = float(np.mean(recalls))
    
    f1s = f1_score(gt_sub, pred_sub, labels = labels, average = None, zero_division =1)
    results['MAP_classification_f1'] = {label_names[labels[i]] : float(f1s[i]) for i in range(len(f1s))}
    results['MAP_classification_f1_macro'] = float(np.mean(f1s))

    return results
  
def get_supervised_scores(gt, pred, label_names, unknown_value=0):
    # gt: 1-dim array of gt labels (per frame)
    # pred: 1-dim array of predicted clusters (per frame)
    # label_names: list of behavior label names
    # returns dict of evaluation scores, assuming we have used a supervised model
    results = {}
    
    ## To avoid issues with macro averaging in sklearn, we have to make sure there are no 'unknowns' that are predicted by the model
    assert unknown_value == 0, "not implemented for unknown value other than 0"
    unknown_shift = (pred == unknown_value).astype(int)
    pred = pred + unknown_shift
    
    ### Get optimized classification scores
    mask = find_unknown_mask(gt, unknown_value = unknown_value)
    gt_sub = gt[mask]
    pred_sub = pred[mask]
    
    labels = np.arange(len(label_names))
    labels = labels[labels != unknown_value]
    
    precisions = precision_score(gt_sub, pred_sub, labels = labels, average = None, zero_division =1)
    results['classification_precision'] = {label_names[labels[i]] : float(precisions[i]) for i in range(len(precisions))}
    results['classification_precision_macro'] = float(np.mean(precisions))
    
    recalls = recall_score(gt_sub, pred_sub, labels = labels, average = None, zero_division =1)
    results['classification_recall'] = {label_names[labels[i]] : float(recalls[i]) for i in range(len(recalls))}
    results['classification_recall_macro'] = float(np.mean(recalls))
    
    f1s = f1_score(gt_sub, pred_sub, labels = labels, average = None, zero_division =1)
    results['classification_f1'] = {label_names[labels[i]] : float(f1s[i]) for i in range(len(f1s))}
    results['classification_f1_macro'] = float(np.mean(f1s))

    return results
  
def mapping_based_scores(gt, pred, num_clusters, label_names, unknown_value = 0, choices = None, probs = None, supervised = False):
    # gt: 1-dim array of gt labels (per frame)
    # pred: 1-dim array of predicted clusters (per frame)
    # label_names: list of behavior label names
    # unknown_value: integer label associated with unknown behavior
    # choices: dict where choices[i] is a list of allowed values to map cluster i to
    # probs: dict where probs[i] is a list of probabilites associated with the choies
    # returns dict of evaluation scores, as well as the dictionaries choices and probs
    
    num_classes = len(label_names)
    
    # Compute choices and probabilities, essentially from confusion matrix
    if choices == None:
      choices, probs = discover_probabilities(gt,
                                              pred,
                                              num_clusters = num_clusters,
                                              num_classes = num_classes,
                                              unknown_value = unknown_value)
    
    mapping_based_score_dict = {}   
    
    if supervised:
      mapping_based_score_dict['supervised_scores'] = get_supervised_scores(gt,
                                                                            pred,
                                                                            label_names,
                                                                            unknown_value=unknown_value,
                                                                           )
    else:
      mapping_based_score_dict['MAP_scores'] = get_MAP_scores(gt,
                                                              pred, 
                                                              choices,
                                                              probs,
                                                              label_names,
                                                              unknown_value = unknown_value, 
                                                             )
    
    return mapping_based_score_dict, choices, probs
    