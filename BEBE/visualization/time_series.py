import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
import pandas as pd

def plot_track(data_fp, predictions_fp, config, eval_dict, start_sample = 0, end_sample = 20000, vars_to_plot = None, target_fp = None):
    input_data = pd.read_csv(data_fp, delimiter = ',', header = None).values #np.load(data_fp)
    metadata = config['metadata']
    sr = metadata['sr']
    clip_column_names = metadata['clip_column_names']
    label_names = metadata['label_names']
    num_labels = len(label_names)
    
    if vars_to_plot is None:
        vars_to_plot = clip_column_names[:-2]
    
    if config['unsupervised']:
      num_rows = len(vars_to_plot) + 3
    else:
      num_rows = len(vars_to_plot) + 2
      
    fig, axes = plt.subplots(nrows=num_rows, ncols=1, figsize=(10, 3* num_rows))
    axes[0].set_title(predictions_fp.split('/')[-1] + ' start: ' + str(start_sample) + ' end: ' + str(end_sample))
    
    # Raw Data
    for i, var in enumerate(vars_to_plot):
        idx = clip_column_names.index(var)
        to_plot = input_data[start_sample: end_sample, idx]
        
        axes[i].set_xlim(left=0, right=(end_sample-start_sample) / sr)
        axes[i].plot(np.arange(len(to_plot))/ float(sr), to_plot, label = var)
        axes[i].set_ylabel(var)
        axes[i].tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            labelbottom=False) # labels along the bottom edge are off    
    
    # Ground truths
    if config['unsupervised']:
      position = -3
    else:
      position = -2
    
    label_idx = clip_column_names.index('label')
    unknown_idx = label_names.index('unknown')
    to_plot = input_data[start_sample: end_sample, label_idx]
    norm = plt.Normalize(0, num_labels)
    axes[position].set_xlim(left=0, right=(end_sample-start_sample) / sr)
    axes[position].scatter(np.arange(len(to_plot))[to_plot!= unknown_idx]/sr, to_plot[to_plot!= unknown_idx], marker = '|', c = to_plot[to_plot!= unknown_idx], cmap = 'Set2', norm = norm, linewidths = 0.1)
    label_ticks = [i for i in range(len(label_names)) if i != unknown_idx]
    axes[position].set_yticks(label_ticks)
    axes[position].set_yticklabels([label_names[i] for i in label_ticks], fontsize = 8, rotation = 45)
    axes[position].set_title("Observed behavior")
    axes[position].tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False) # labels along the bottom edge are off

    # Plot predictions
    class_predictions = pd.read_csv(predictions_fp, delimiter = ',', header = None).values.flatten() 
    
    if config['unsupervised']:
      # Plot discovered clusters
      to_plot = class_predictions[start_sample: end_sample]
      axes[-2].set_xlim(left=0, right=(end_sample-start_sample)/ sr)
      axes[-2].scatter(np.arange(len(to_plot))/sr, to_plot, marker = '|', c = to_plot, cmap = 'hsv', linewidths = 0.1)
      axes[-2].set_ylabel("Discovered motif number")

      axes[-2].set_ylim(bottom=-0.5, top = config['num_clusters']-0.5)
      major_tick_spacing = max(1, config['num_clusters'] // 8)
      axes[-2].yaxis.set_major_locator(MultipleLocator(major_tick_spacing))
      axes[-2].yaxis.set_minor_locator(MultipleLocator(1))


      axes[-2].set_title("Discovered behavioral motifs")
      axes[-2].tick_params(
          axis='x',          # changes apply to the x-axis
          which='both',      # both major and minor ticks are affected
          bottom=False,      # ticks along the bottom edge are off
          top=False,         # ticks along the top edge are off
          labelbottom=False) # labels along the bottom edge are off
    
      # Max a posteriori assignment clusters -> labels
      mapping_dict = eval_dict['MAP_scores']['MAP_mapping_dict']
      to_plot = class_predictions[start_sample: end_sample]
      to_plot = list(to_plot)
      to_plot = np.array(list(map(lambda x : mapping_dict[x], to_plot)))
      axes[-1].set_xlim(left=0, right=(end_sample-start_sample) /sr)
      axes[-1].scatter(np.arange(len(to_plot))/sr, to_plot, marker = '|', c = to_plot, cmap = 'Set2', norm = norm, linewidths = 0.1)
      label_ticks = [i for i in range(len(label_names)) if i != unknown_idx]
      axes[-1].set_yticks(label_ticks)
      axes[-1].set_yticklabels([label_names[i] for i in label_ticks], fontsize = 8, rotation = 45)
      axes[-1].set_title("Model prediction (a posteriori assignment of discovered motifs to behavior labels)")
      axes[-1].set_xlabel("Time (seconds)")
      
    else:
      # Just need to plot predicted class labels
      # Max a posteriori assignment clusters -> labels
      to_plot = class_predictions[start_sample: end_sample]
      axes[-1].set_xlim(left=0, right=(end_sample-start_sample) /sr)
      axes[-1].scatter(np.arange(len(to_plot))/sr, to_plot, marker = '|', c = to_plot, cmap = 'Set2', norm = norm, linewidths = 0.1)
      label_ticks = [i for i in range(len(label_names)) if i != unknown_idx]
      axes[-1].set_yticks(label_ticks)
      axes[-1].set_yticklabels([label_names[i] for i in label_ticks], fontsize = 8, rotation = 45)
      axes[-1].set_title("Model prediction")
      axes[-1].set_xlabel("Time (seconds)")
      
    if target_fp is not None:
        plt.savefig(target_fp); plt.close()
    
    else:
        plt.show()