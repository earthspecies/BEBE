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
    array = np.array(array)
    # in: 1-dim array
    # out: 1-dim boolean array with 0's anywhere that is within tolerance_frames frames of a frame containing unknown_value
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
    # gt 1-dim array of gt labels (per frame)
    # pred 1-dim array of predicted clusters (per frame)
    # num_clusters number of clusters allowed, we assume the clusters are numbered 0,1,2,...   
    # num_classes number of classes allowed, includes unknown, assume numbered 0,1,2,...
    # Returns:
    # choices, dict where choices[i] is a list of allowed values to map cluster i to
    # probs, dict where probs[i] is a list of probabilites associated with the choies
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
    mapping_dict = {}
    for i in choices:
        best_idx = np.argmax(probs[i])
        mapping_dict[int(i)] = int(choices[i][best_idx]) # cast as int for saving off results
    return lambda i : mapping_dict[i], mapping_dict
  
def produce_random_cluster_to_label(choices, probs):
    mapping_dict = {}
    for i in choices:
        mapping_dict[i] = np.random.choice(choices[i], p = probs[i])
    return lambda i: mapping_dict[i], mapping_dict
  
def find_boundaries(array):
    # in: 1-dim array
    # out: 1-dim boolean array arr_bound
    # with arr_bound[i] = True iff array[i] != array [i+1]
    # and arr_bound[len(array)-1] = True
    arr_bound = (array[:-1] - array[1:]) != 0
    arr_bound = np.append(arr_bound, [True])
    return arr_bound
  
### Boundary precision and recall:
### nb counting hits is assymmetric

def boundary_precision_with_unknown_and_tolerance(gt_bound, pred_bound, gt_mask, tolerance_frames = 0):
    eps = 1e-6
    boundary_alignments = []
    boundary_alignments.append(pred_bound * gt_bound)
    for shift in range(1, tolerance_frames + 1):
        right_shifted_gt_bound = gt_bound[shift:]
        right_shifted_gt_bound = np.append(right_shifted_gt_bound, np.full(shift, False))
        left_shifted_gt_bound = gt_bound[:-shift]
        left_shifted_gt_bound = np.append(np.full(shift, False), left_shifted_gt_bound)
        boundary_alignments.append(pred_bound * right_shifted_gt_bound)
        boundary_alignments.append(pred_bound * left_shifted_gt_bound)

    hit_array = sum(boundary_alignments).astype(bool)
    hit_array = hit_array[gt_mask]
    return np.sum(hit_array) / (np.sum(pred_bound[gt_mask]) + eps)

def boundary_recall_with_unknown_and_tolerance(gt_bound, pred_bound, gt_mask, tolerance_frames = 0):
    eps = 1e-6
    boundary_alignments = []
    boundary_alignments.append(pred_bound * gt_bound)
    for shift in range(1, tolerance_frames + 1):
        right_shifted_pred_bound = pred_bound[shift:]
        right_shifted_pred_bound = np.append(right_shifted_pred_bound, np.full(shift, False))
        left_shifted_pred_bound = pred_bound[:-shift]
        left_shifted_pred_bound = np.append(np.full(shift, False), left_shifted_pred_bound)
        boundary_alignments.append(gt_bound * right_shifted_pred_bound)
        boundary_alignments.append(gt_bound * left_shifted_pred_bound)

    hit_array = sum(boundary_alignments).astype(bool)
    hit_array = hit_array[gt_mask]
    return np.sum(hit_array) / (np.sum(gt_bound[gt_mask]) + eps)
  
def compute_f1(prec, rec):
    eps = 1e-6
    return (2* prec*rec) / (prec+rec + eps)

def compute_R(prec, rec):
    # uses the formula for oversegmentation from Self-Supervised Contrastive Learning for Unsupervised Phoneme Segmentation by Kreuk et al
    # note this doesn't follow the definition in the original in An Improved Speech Segmentation Quality Measure: the R-value by Rasanen et al
    # because of the assymetry in the numerator of precision and recall
    # Then again, our definition of precision and recall don't follow Rasanen either so perhaps the differences cancel out
    # Anyways, I follow what seems to be the modern standard
    eps = 1e-12
    over_segmentation = rec/(prec + eps) - 1
    r1 = np.sqrt((1-rec)**2 + over_segmentation**2)
    r2 = (rec - over_segmentation - 1)/np.sqrt(2.)
    return (2 - np.abs(r1) - np.abs(r2))/2.
  
def compute_single_score_randomized(choices, probs, pred, mask, gt_sub, gt_bound, gt_mask_boundaries, boundary_tolerance_frames):
  # sample mapping
  #print('producing mapping')
  _, mapping_dict = produce_random_cluster_to_label(choices, probs)
  #print('produced')
  # pred_list = list(pred)
  # #pred_mapped = np.array(list(map(mapping, pred_list)))
  # pred_mapped = map(mapping, pred_list)
  # pred_mapped = np.fromiter(pred_mapped, int, count = len(pred_list))
  
  ###
  
  outs = []
  for i in mapping_dict:
    # mostly vectorized way to assign i \mapsto mapping_dict[i]
    m = pred == i
    outs.append(m * mapping_dict[i])
  pred_mapped = sum(outs)
  
  ###
  
  
  #print('cast')
  pred_sub = pred_mapped[mask]
  #print('masked')
  prec = precision_score(gt_sub, pred_sub, average = 'macro', zero_division =0 )
  rec = recall_score(gt_sub, pred_sub, average = 'macro', zero_division =0 )
  f1 = f1_score(gt_sub, pred_sub, average = 'macro', zero_division =0 )
  #print('scored class')

  pred_bound = find_boundaries(pred_mapped)
  bprec = boundary_precision_with_unknown_and_tolerance(gt_bound,
                                                        pred_bound,
                                                        gt_mask_boundaries,
                                                        tolerance_frames = boundary_tolerance_frames
                                                       )
  brec = boundary_recall_with_unknown_and_tolerance(gt_bound,
                                                    pred_bound,
                                                    gt_mask_boundaries,
                                                    tolerance_frames = boundary_tolerance_frames
                                                   )
  bf1 = compute_f1(bprec, brec)
  bR = compute_R(bprec, brec)
  #print('scored boundary')
  return prec, rec, f1, bprec, brec, bf1, bR
  
def estimate_averaged_scores(gt, pred, choices, probs, unknown_value=0, boundary_tolerance_frames = 0, n_iter = 1):
    # For each cluster, samples a label according to proportion of overlap
    # Maps that cluster to that label, and computes precision, recall, etc
    # Does this n_iter times and computes the average
    # (finding the exact value would be intractable)
    # Returns a dict including the MAP mapping from clusters to labels
    results = {}
    ### Estimate average classification scores
    mask = find_unknown_mask(gt, unknown_value = unknown_value)
    gt_sub = gt[mask]
    #pred_list = list(pred)
    
    ### Also estimate average boundary scores
    gt_bound = find_boundaries(gt)
    # need to shift the mask to the right, since we are looking at boundaries between 2 frames:
    gt_mask = find_unknown_mask(gt, tolerance_frames = boundary_tolerance_frames)
    gt_mask_boundaries = gt_mask * np.append(gt_mask[1:], [False]) 
    gt_mask_boundaries = gt_mask_boundaries.astype(bool)
    
    ### initialize
    prec = []
    rec = []
    f1 = []
    boundary_prec = []
    boundary_rec = []
    boundary_f1 = []
    boundary_R = []
    
    
    print("Sampling to estimate averaged mapping based scores")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers = os.cpu_count() + 4) as executor:
      futures = list(tqdm.tqdm(executor.map(compute_single_score_randomized,
                                            itertools.repeat(choices, n_iter),
                                            itertools.repeat(probs, n_iter),
                                            itertools.repeat(pred, n_iter), 
                                            itertools.repeat(mask, n_iter), 
                                            itertools.repeat(gt_sub, n_iter), 
                                            itertools.repeat(gt_bound, n_iter),
                                            itertools.repeat(gt_mask_boundaries, n_iter),
                                            itertools.repeat(boundary_tolerance_frames, n_iter)), total = n_iter))
    for future in futures:
      single_prec, single_rec, single_f1, single_bprec, single_brec, single_bf1, single_bR = future
      prec.append(single_prec)
      rec.append(single_rec)
      f1.append(single_f1)
      boundary_prec.append(single_bprec)
      boundary_rec.append(single_brec)
      boundary_f1.append(single_bf1)
      boundary_R.append(single_bR)
    
    
    ###first attempt to paralellize, doesn't work
    
#     with tqdm.tqdm(total=n_iter) as pbar:
    
#       with concurrent.futures.ProcessPoolExecutor() as executor:
#           futures = [executor.submit(compute_single_score_randomized, choices, probs, pred_list, mask, gt_sub, gt_bound, gt_mask_boundaries, boundary_tolerance_frames) for _ in range(n_iter)]

#           for future in concurrent.futures.as_completed(futures):
#               single_prec, single_rec, single_f1, single_bprec, single_brec, single_bf1, single_bR = future.result()
#               prec.append(single_prec)
#               rec.append(single_rec)
#               f1.append(single_f1)
#               boundary_prec.append(single_bprec)
#               boundary_rec.append(single_brec)
#               boundary_f1.append(single_bf1)
#               boundary_R.append(single_bR)
#               pbar.update(1)
      
#     Non-parallel version
#     for i in tqdm.tqdm(range(n_iter)):
#         # sample mapping
#         mapping, _ = produce_random_cluster_to_label(choices, probs)
#         pred_mapped = np.array(list(map(mapping, pred_list)))
#         pred_sub = pred_mapped[mask]
#         prec.append(precision_score(gt_sub, pred_sub, average = 'macro', zero_division =0 ))
#         rec.append(recall_score(gt_sub, pred_sub, average = 'macro', zero_division =0 ))
#         f1.append(f1_score(gt_sub, pred_sub, average = 'macro', zero_division =0 ))
        
#         pred_bound = find_boundaries(pred_mapped)
#         bprec = boundary_precision_with_unknown_and_tolerance(gt_bound,
#                                                               pred_bound,
#                                                               gt_mask_boundaries,
#                                                               tolerance_frames = boundary_tolerance_frames
#                                                              )
#         boundary_prec.append(bprec)
#         brec = boundary_recall_with_unknown_and_tolerance(gt_bound,
#                                                           pred_bound,
#                                                           gt_mask_boundaries,
#                                                           tolerance_frames = boundary_tolerance_frames
#                                                          )
#         boundary_rec.append(brec)
#         boundary_f1.append(compute_f1(bprec, brec))
#         boundary_R.append(compute_R(bprec, brec))
###
        
    results['averaged_classification_precision'] = np.mean(prec)
    results['averaged_classification_recall'] = np.mean(rec)
    results['averaged_classification_f1'] = np.mean(f1)
    results['averaged_boundary_precision'] = np.mean(boundary_prec)
    results['averaged_boundary_recall'] = np.mean(boundary_rec)
    results['averaged_boundary_f1'] = np.mean(boundary_f1)
    results['averaged_boundary_R'] = np.mean(boundary_R)
    for key in results:
      results[key] = float(results[key])    
    return results
  
def get_MAP_scores(gt, pred, choices, probs, unknown_value=0, boundary_tolerance_frames = 0):
    # For each cluster, looks for the label with highest overlap with that cluster
    # (i.e. the Maximimum a posteriori estimate)
    # Maps that cluster to that label, and computes precision, recall, etc
    # Returns a dict including the MAP mapping from clusters to labels
    results = {}
    pred_list = list(pred)
    mapping, mapping_dict = produce_MAP_cluster_to_label(choices, probs)
    results['MAP_mapping_dict'] = mapping_dict    
    pred_mapped = np.array(list(map(mapping, pred_list)))
    
    ### Get optimized classification scores
    mask = find_unknown_mask(gt, unknown_value = unknown_value)
    gt_sub = gt[mask]
    pred_sub = pred_mapped[mask]
    
    results['MAP_classification_precision'] = float(precision_score(gt_sub, pred_sub, average = 'macro', zero_division =0))
    results['MAP_classification_recall'] = float(recall_score(gt_sub, pred_sub, average = 'macro', zero_division =0))
    results['MAP_classification_f1'] = float(f1_score(gt_sub, pred_sub, average = 'macro', zero_division =0))
    
    ### Get optimized segmentation boundary scores    
    gt_bound = find_boundaries(gt)
    # need to shift the mask to the right, since we are looking at boundaries between 2 frames:
    gt_mask = find_unknown_mask(gt, tolerance_frames = boundary_tolerance_frames)
    gt_mask_boundaries = gt_mask * np.append(gt_mask[1:], [False]) 
    gt_mask_boundaries = gt_mask_boundaries.astype(bool)
    
    pred_bound = find_boundaries(pred_mapped)
    results['MAP_boundary_precision'] = float(boundary_precision_with_unknown_and_tolerance(gt_bound, 
                                                                                            pred_bound, 
                                                                                            gt_mask_boundaries, 
                                                                                            tolerance_frames = boundary_tolerance_frames
                                                                                           ))
    results['MAP_boundary_recall'] = float(boundary_recall_with_unknown_and_tolerance(gt_bound,
                                                                                      pred_bound,
                                                                                      gt_mask_boundaries,
                                                                                      tolerance_frames = boundary_tolerance_frames
                                                                                     ))
    results['MAP_boundary_f1'] = float(compute_f1(results['MAP_boundary_precision'], results['MAP_boundary_recall']))
    results['MAP_boundary_R'] = float(compute_R(results['MAP_boundary_precision'], results['MAP_boundary_recall']))
    return results
  
def mapping_based_scores(gt, pred, num_clusters, num_classes, boundary_tolerance_frames = 0, unknown_value = 0, choices = None, probs = None, n_samples = 100):
    # Main function to produce mapping based scores
    
    # Compute choices and probabilities, essentially from confusion matrix
    if choices == None:
      choices, probs = discover_probabilities(gt,
                                              pred,
                                              num_clusters = num_clusters,
                                              num_classes = num_classes,
                                              unknown_value = unknown_value)
    
    mapping_based_score_dict = {}
    
    if n_samples > 0:
      mapping_based_score_dict['averaged_scores'] = estimate_averaged_scores(gt, 
                                                                             pred, 
                                                                             choices, 
                                                                             probs, 
                                                                             unknown_value=unknown_value, 
                                                                             boundary_tolerance_frames = boundary_tolerance_frames, 
                                                                             n_iter = n_samples)
    else:
      mapping_based_score_dict['averaged_scores'] = None
    
    mapping_based_score_dict['MAP_scores'] = get_MAP_scores(gt,
                                                            pred, 
                                                            choices, 
                                                            probs, 
                                                            unknown_value = unknown_value, 
                                                            boundary_tolerance_frames = boundary_tolerance_frames)
    
    return mapping_based_score_dict, choices, probs
    