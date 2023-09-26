# assumes run from parent directory

from plumbum import local, FG
import os
import yaml
import numpy as np


def main():
    # Harnet ablations
    
    model_types = ['harnet', 'harnet_unfrozen', 'harnet_random', 'RNN', 'CRNN'] # 'rf']
    low_data = [True, False]
    dataset_names = ['baglione_crows',
                     'jeantet_turtles',
                     'pagano_bears',
                     'maekawa_gulls',
                     'HAR',
                     'friedlaender_whales',
                     'ladds_seals',
                     'vehkaoja_dogs',
                     'desantis_rattlesnakes',
                    ]
  
    for model_type in model_types:
        for low_data_setting in low_data:
            for dataset_name in dataset_names:
                if low_data_setting:
                    experiment_name = f'{dataset_name}_{model_type}_acc_and_depth_only'
                    local['python']['cross_val_experiment.py',
                                    '--experiment-dir-parent=/home/jupyter/behavior_benchmarks_experiments',
                                    f'--experiment-name={experiment_name}',
                                    f'--dataset-dir=/home/jupyter/behavior_data_local/{dataset_name}',
                                    f'--model={model_type}',
                                    '--acc-and-depth-only',
                                    '--no-cutoff',
                                    '--resume',
                                    '--low-data-setting'
                                   ] & FG
                    
                    
                    
                else:
                    experiment_name = f'{dataset_name}_{model_type}_low_data_acc_and_depth_only'
                    local['python']['cross_val_experiment.py',
                                    '--experiment-dir-parent=/home/jupyter/behavior_benchmarks_experiments',
                                    f'--experiment-name={experiment_name}',
                                    f'--dataset-dir=/home/jupyter/behavior_data_local/{dataset_name}',
                                    f'--model={model_type}',
                                    '--acc-and-depth-only',
                                    '--no-cutoff',
                                    '--resume'
                                   ] & FG
                    
    # Low data ablations (with all data channels)
    
    model_types = ['CRNN'] # 'rf']
    low_data = [True, False]
    dataset_names = ['baglione_crows',
                     'jeantet_turtles',
                     'pagano_bears',
                     'maekawa_gulls',
                     'HAR',
                     'friedlaender_whales',
                     'ladds_seals',
                     'vehkaoja_dogs',
                     'desantis_rattlesnakes',
                    ]
  
    for model_type in model_types:
        for low_data_setting in low_data:
            for dataset_name in dataset_names:
                if low_data_setting:
                    experiment_name = f'{dataset_name}_{model_type}'
                    local['python']['cross_val_experiment.py',
                                    '--experiment-dir-parent=/home/jupyter/behavior_benchmarks_experiments',
                                    f'--experiment-name={experiment_name}',
                                    f'--dataset-dir=/home/jupyter/behavior_data_local/{dataset_name}',
                                    f'--model={model_type}',
                                    '--no-cutoff',
                                    '--resume',
                                    '--low-data-setting'
                                   ] & FG
                    
                    
                    
                else:
                    experiment_name = f'{dataset_name}_{model_type}_low_data'
                    local['python']['cross_val_experiment.py',
                                    '--experiment-dir-parent=/home/jupyter/behavior_benchmarks_experiments',
                                    f'--experiment-name={experiment_name}',
                                    f'--dataset-dir=/home/jupyter/behavior_data_local/{dataset_name}',
                                    f'--model={model_type}',
                                    '--no-cutoff',
                                    '--resume'
                                   ] & FG               
                    
    

if __name__ == "__main__":
    main()
  