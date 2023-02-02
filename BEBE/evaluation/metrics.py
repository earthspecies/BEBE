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
  
def contingency_analysis(gt, pred, num_clusters, num_classes, unknown_value = 0):
    # gt: 1-dim array of integers, ground truth labels (per frame)
    # pred: 1-dim array of integers, predicted clusters (per frame)
    # num_clusters: number of clusters allowed, we assume the clusters are numbered 0,1,2,...   
    # num_classes: number of classes allowed, includes unknown, assume numbered 0,1,2,...
    
    mask = find_unknown_mask(gt, unknown_value = unknown_value)
    gt_sub = gt[mask]
    pred_sub = pred[mask]
    
    # Computes a (sparse) contingency matrix. Look for max value in each column
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
            
    mapping_dict = {}
    for i in choices:
        best_idx = np.argmax(probs[i])
        mapping_dict[int(i)] = int(choices[i][best_idx])
        
    return mapping_dict
                                          
def get_time_scale_ratio(pred, target_time_scale_sec, sr):
    n_label_bouts = sum(pred[1:] != pred[:-1]) + 1
    mean_dur_samples = len(pred) / n_label_bouts
    mean_dur_sec = mean_dur_samples / sr
    time_scale_ratio = mean_dur_sec / target_time_scale_sec
    return float(time_scale_ratio)

def get_unsupervised_scores(gt, pred, mapping_dict, label_names, unknown_value=0, target_time_scale_sec = 1., sr = 1.):
    # For each cluster, looks for the label with highest overlap with that cluster
    # Maps that cluster to that label, and computes precision, recall, etc
    # Returns: dict including the MAP mapping from clusters to labels
    # gt: 1-dim array of gt labels (per frame)
    # pred: 1-dim array of predicted clusters (per frame)
    # mapping_dict: cluster to label mapping discovered by contingency analysis
    # label_names: list of behavior label names
    results = {}
    pred_list = list(pred)
    results['contingency_analysis_mapping_dict'] = mapping_dict    
    mapping = lambda i : mapping_dict[i]
    pred_mapped = np.array(list(map(mapping, pred_list)))
    
    ### Get classification scores
    mask = find_unknown_mask(gt, unknown_value = unknown_value)
    gt_sub = gt[mask]
    pred_sub = pred_mapped[mask]
    
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
    
    results['time_scale_ratio'] = get_time_scale_ratio(pred_mapped, target_time_scale_sec, sr)  
    return results
  
def get_supervised_scores(gt, pred, label_names, unknown_value=0, target_time_scale_sec = 1., sr = 1.):
    # gt: 1-dim array of integer gt labels (per frame)
    # pred: 1-dim array of integer predicted clusters (per frame)
    # label_names: list of str behavior label names
    # returns dict of evaluation scores
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
    
    results['time_scale_ratio'] = get_time_scale_ratio(pred, target_time_scale_sec, sr)
    return results
  
def mapping_based_scores(gt, pred, num_clusters, label_names, unknown_value = 0, mapping_dict = None, supervised = False, target_time_scale_sec = 1., sr = 1.):
    # gt: 1-dim array of int gt labels (per frame)
    # pred: 1-dim array of int predicted clusters (per frame)
    # label_names: list of str behavior label names
    # unknown_value: integer label associated with unknown behavior (default 0)
    # mapping_dict: dict where mapping_dict[i] the behavior label index that cluster i is sent to under contingency analysis
    # returns dict of evaluation scores, as well as mapping dict
    
    num_classes = len(label_names)
    
    # Compute mapping dict if None
    if mapping_dict == None:
      mapping_dict = contingency_analysis(gt,
                                          pred,
                                          num_clusters = num_clusters,
                                          num_classes = num_classes,
                                          unknown_value = unknown_value)
    
    if supervised:
      mapping_based_score_dict = get_supervised_scores(gt,
                                                       pred,
                                                       label_names,
                                                       unknown_value=unknown_value,
                                                       target_time_scale_sec = target_time_scale_sec,
                                                       sr = sr
                                                      )
    else:
      mapping_based_score_dict = get_unsupervised_scores(gt,
                                                         pred,
                                                         mapping_dict,
                                                         label_names,
                                                         unknown_value = unknown_value, 
                                                         target_time_scale_sec = target_time_scale_sec,
                                                         sr = sr
                                                        )
    
    return mapping_based_score_dict, mapping_dict
    