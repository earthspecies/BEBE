import numpy as np
from matplotlib import pyplot as plt

def plot_track(data_fp, predictions_fp, config, start_sample = 0, end_sample = 20000, vars_to_plot = None, target_fp = None):
    input_data = np.load(data_fp)
    metadata = config['metadata']
    clip_column_names = metadata['clip_column_names']
    label_names = metadata['label_names']
    
    if vars_to_plot is None:
        vars_to_plot = clip_column_names[:-2]
    
    num_rows = len(vars_to_plot) + 2
    fig, axes = plt.subplots(nrows=num_rows, ncols=1, figsize=(10, 2* num_rows))
    #fig = plt.figure(figsize = (10, 2* num_rows))
    axes[0].set_title(predictions_fp.split('/')[-1] + ' start: ' + str(start_sample) + ' end: ' + str(end_sample))
    
    # Raw Data
    for i, var in enumerate(vars_to_plot):
        idx = clip_column_names.index(var)
        to_plot = input_data[start_sample: end_sample, idx]
        
        #ax = fig.add_subplot(num_rows,1,i+1)
        axes[i].set_xlim(left=0, right=end_sample-start_sample)
        axes[i].plot(to_plot, label = var)
        axes[i].set_ylabel(var)
        #plt.legend()

    # Ground truths
    label_idx = clip_column_names.index('label')
    unknown_idx = label_names.index('unknown')
    to_plot = input_data[start_sample: end_sample, label_idx]
    axes[-2].set_xlim(left=0, right=end_sample-start_sample)
    axes[-2].scatter(np.arange(len(to_plot))[to_plot!= unknown_idx], to_plot[to_plot!= unknown_idx], marker = '|', c = to_plot[to_plot!= unknown_idx], cmap = 'Set2')
    label_ticks = [i for i in range(len(label_names)) if i != unknown_idx]
    axes[-2].set_yticks(label_ticks)
    axes[-2].set_yticklabels([label_names[i] for i in label_ticks])
    axes[-2].set_ylabel("ground truth")

    # Clusters
    clusters_data = np.load(predictions_fp)
    to_plot = clusters_data[start_sample: end_sample]
    axes[-1].set_xlim(left=0, right=end_sample-start_sample)
    axes[-1].scatter(np.arange(len(to_plot)), to_plot, marker = '|', c = to_plot, cmap = 'hsv')
    axes[-1].set_ylabel("cluster")

    if target_fp is not None:
        plt.savefig(target_fp)
    
    else:
        plt.show()