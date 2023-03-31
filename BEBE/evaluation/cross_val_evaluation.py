import yaml
import numpy as np
from pathlib import Path
import seaborn as sns
from matplotlib import pyplot as plt

def cross_val_evaluation(final_experiment_dir, metadata):
  label_names = metadata['label_names'][1:]
  results = {}

  for split in ['test', 'train']:
    eval_fps = list(Path(final_experiment_dir).glob(f'**/{split}_eval.yaml'))
    if len(eval_fps)>0:
      results[split] = {}
      f1s = []
      precs = []
      recs = []
      tsrs = []
      ground_truth_label_counts = {label : 0 for label in label_names}
      predicted_label_counts = {label : 0 for label in label_names}
      
      for fp in eval_fps:
          with open(fp, 'r') as f:
              x = yaml.safe_load(f)
          f1s.extend(x['individual_scores']['macro_f1s'])
          precs.extend(x['individual_scores']['macro_precisions'])
          recs.extend(x['individual_scores']['macro_recalls'])
          tsrs.extend(x['individual_scores']['time_scale_ratios'])
          for label in label_names:
            ground_truth_label_counts[label] += x['overall_scores']['ground_truth_label_counts'][label]
            predicted_label_counts[label] += x['overall_scores']['predicted_label_counts'][label]
            
      print(f"{split} f1  : %1.3f (%1.3f)" % (np.mean(f1s), np.std(f1s)))
      print(f"{split} Prec: %1.3f (%1.3f)" % (np.mean(precs), np.std(precs)))
      print(f"{split} Rec : %1.3f (%1.3f)" % (np.mean(recs), np.std(recs)))
      print(F"{split} TSR : %1.3f (%1.3f)" % (np.mean(tsrs), np.std(tsrs)))
      results[split]['f1_mean'] = float(np.mean(f1s))
      results[split]['f1_std'] = float(np.std(f1s))
      results[split]['prec_mean'] = float(np.mean(precs))
      results[split]['prec_std'] = float(np.std(precs))
      results[split]['rec_mean'] = float(np.mean(recs))
      results[split]['rec_std'] = float(np.std(recs))
      results[split]['tsr_mean'] = float(np.mean(tsrs))
      results[split]['tsr_std'] = float(np.std(tsrs))
      results[split]['ground_truth_label_counts'] = ground_truth_label_counts
      results[split]['predicted_label_counts'] = predicted_label_counts
      
    individualized_eval_fps = list(Path(final_experiment_dir).glob(f'**/{split}_f1_consistency.yaml'))
    if len(individualized_eval_fps)>0:
      f1s = []
      for fp in individualized_eval_fps:
        with open(fp, 'r') as f:
              x = yaml.safe_load(f)
        f1s.extend(x['macro_f1s_individualized'])
      results[f"{split}_f1_individualized_mean"] = float(np.mean(f1s))
      results[f"{split}_f1_individualized_std"] = float(np.std(f1s))
      print(f"{split} individualized f1 : %1.3f (%1.3f)" % (np.mean(f1s), np.std(f1s)))
      
    cm_fps = list(Path(final_experiment_dir).glob(f'**/{split}_confusion_matrix_for_xval.npy'))
    if len(cm_fps)>0:
      cm = None
      for cm_fp in cm_fps:
        single_cm = np.load(cm_fp)
        if cm is None:
          cm=single_cm
        else:
          cm += single_cm
      confusion_matrix(cm, label_names, final_experiment_dir, name=f"{metadata['dataset_name']}_{split}")

  target_fp = Path(final_experiment_dir, 'final_result_summary.yaml')                       
  with open(target_fp, 'w') as file:
      yaml.dump(results, file)
      
      
def confusion_matrix(data, label_names, target_dir, name=""):
    data_normalized = data / np.sum(data)
    
    fig = plt.figure(num=None, figsize=(12, 8), dpi=80, facecolor='w', edgecolor='k')
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    sns.heatmap(data_normalized, annot=True, fmt='.3f', cmap = 'magma', cbar = True, vmin = 0, vmax = 1, ax = ax)
    ax.set_title(f'Proportion of {name} motion behavioral states (rows sum to 1)')
    ax.set_yticks([i + 0.5 for i in range(len(label_names))])
    ax.set_yticklabels(label_names, rotation = 0)
    ax.set_xticks([i + 0.5 for i in range(len(label_names))])
    ax.set_xticklabels(label_names, rotation = -15)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('Behavior Label')
    plt.title(name)
    
    plt.savefig(Path(target_dir, f"{name}_confusion_matrix.png"))