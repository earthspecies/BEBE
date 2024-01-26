# assumes run from parent directory

from plumbum import local, FG
import os
import yaml
import numpy as np


def main():
    # Harnet ablations
    
    model_types = ['CRNN', 'harnet', 'harnet_unfrozen', 'harnet_random', 'RNN', 'rf']
    low_data = [False, True]
    dataset_names = ['maekawa_gulls']
    # dataset_names = ['baglione_crows',
    #                  'jeantet_turtles',
    #                  'pagano_bears',
    #                  'maekawa_gulls',
    #                  'HAR',
    #                  'friedlaender_whales',
    #                  'ladds_seals',
    #                  'vehkaoja_dogs',
    #                  'desantis_rattlesnakes',
    #                 ]
  
    for model_type in model_types:
        for low_data_setting in low_data:
            for dataset_name in dataset_names:
                if low_data_setting:
                    experiment_name = f'{dataset_name}_{model_type}_low_data_nogyr'
                    local['python']['cross_val_experiment.py',
                                    '--experiment-dir-parent=/home/jupyter/behavior_benchmarks_experiments',
                                    f'--experiment-name={experiment_name}',
                                    f'--dataset-dir=/home/jupyter/behavior_data_local/{dataset_name}',
                                    f'--model={model_type}',
                                    '--nogyr',
                                    '--no-cutoff',
                                    '--resume',
                                    '--low-data-setting'
                                   ] & FG
                    
                    
                    
                else:
                    experiment_name = f'{dataset_name}_{model_type}_nogyr'
                    local['python']['cross_val_experiment.py',
                                    '--experiment-dir-parent=/home/jupyter/behavior_benchmarks_experiments',
                                    f'--experiment-name={experiment_name}',
                                    f'--dataset-dir=/home/jupyter/behavior_data_local/{dataset_name}',
                                    f'--model={model_type}',
                                    '--nogyr',
                                    '--no-cutoff',
                                    '--resume'
                                   ] & FG
                    
    # Harnet ablations: wavelets, where we do use cutoff frequency as a hyperparam
                    
    model_types = ['wavelet_RNN']
    low_data = [True] # False: hyperparam selection takes too long
    dataset_names = ['maekawa_gulls']

    # dataset_names = ['baglione_crows',
    #                  'jeantet_turtles',
    #                  'pagano_bears',
    #                  'maekawa_gulls',
    #                  'HAR',
    #                  'friedlaender_whales',
    #                  'ladds_seals',
    #                  'vehkaoja_dogs',
    #                  'desantis_rattlesnakes',
    #                 ]
  
    for model_type in model_types:
        for low_data_setting in low_data:
            for dataset_name in dataset_names:
                if low_data_setting:
                    experiment_name = f'{dataset_name}_{model_type}_low_data_nogyr'
                    local['python']['cross_val_experiment.py',
                                    '--experiment-dir-parent=/home/jupyter/behavior_benchmarks_experiments',
                                    f'--experiment-name={experiment_name}',
                                    f'--dataset-dir=/home/jupyter/behavior_data_local/{dataset_name}',
                                    f'--model={model_type}',
                                    '--nogyr',
                                    '--resume',
                                    '--low-data-setting'
                                   ] & FG
                    
                    
                    
                else:
                    experiment_name = f'{dataset_name}_{model_type}_nogyr'
                    local['python']['cross_val_experiment.py',
                                    '--experiment-dir-parent=/home/jupyter/behavior_benchmarks_experiments',
                                    f'--experiment-name={experiment_name}',
                                    f'--dataset-dir=/home/jupyter/behavior_data_local/{dataset_name}',
                                    f'--model={model_type}',
                                    '--nogyr',
                                    '--resume'
                                   ] & FG
                    
    # Low data ablations (with all data channels)
    
    model_types = ['CRNN', 'rf'] 
    low_data = [True] # false is already done in general experiments
    dataset_names = ['maekawa_gulls']

    # dataset_names = ['baglione_crows',
    #                  'jeantet_turtles',
    #                  'pagano_bears',
    #                  'maekawa_gulls',
    #                  'HAR',
    #                  'friedlaender_whales',
    #                  'ladds_seals',
    #                  'vehkaoja_dogs',
    #                  'desantis_rattlesnakes',
    #                 ]
  
    for model_type in model_types:
        for low_data_setting in low_data:
            for dataset_name in dataset_names:
                if low_data_setting:
                    experiment_name = f'{dataset_name}_{model_type}_low_data'
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
                    experiment_name = f'{dataset_name}_{model_type}'
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
  