import os
import hydra
import torch
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import KFold

from src.train import train, eval
from src.plot_utils import plot_adj_matrices
from src.dataset import load_dataset


@hydra.main(version_base="1.3.2", config_path="configs", config_name="experiment")
def main(config):
    torch.cuda.empty_cache()

    if torch.cuda.is_available():
        print("Running on GPU")
    else:
        print("Running on CPU")

    kf = KFold(n_splits=config.experiment.kfold.n_splits, 
               shuffle=config.experiment.kfold.shuffle, 
               random_state=config.experiment.kfold.random_state)

    # Initialize folder structure for this run
    base_dir = config.experiment.base_dir
    model_name = config.model.name
    dataset_type = config.dataset.name
    run_name = config.experiment.run_name
    run_dir = f'{base_dir}/{model_name}/{dataset_type}/{run_name}/'

    # Load dataset
    source_data, target_data = load_dataset(config)


    for fold, (train_idx, val_idx) in enumerate(kf.split(source_data)):
        print(f"Training Fold {fold+1}/3")

        # Initialize results directory
        res_dir = f'{run_dir}fold_{fold+1}/'
        if not os.path.exists(res_dir):
            os.makedirs(res_dir)

        # Fetch training and val data for this fold
        source_data_train = [source_data[i] for i in train_idx]
        target_data_train = [target_data[i] for i in train_idx]
        source_data_val = [source_data[i] for i in val_idx]
        target_data_val = [target_data[i] for i in val_idx]

        # Train model for this fold
        train_output = train(config, 
                              source_data_train, 
                              target_data_train,
                              source_data_val,
                              target_data_val, 
                              res_dir)

        # Evaluate model for this fold
        eval_output, eval_loss = eval(config, 
                                      train_output['model'], 
                                      source_data_val, 
                                      target_data_val, 
                                      train_output['dual_pyg'], 
                                      train_output['critereon'])

        # Final evaluation loss for this fold
        print(f"Final Validation Loss (Target): {eval_loss}")

        # Save source, taregt, and eval output for this fold
        np.save(f'{res_dir}/eval_output.npy', np.array(eval_output))
        np.save(f'{res_dir}/source.npy', np.array([s['mat'] for s in source_data_val]))
        np.save(f'{res_dir}/target.npy', np.array([t['mat'] for t in target_data_val]))


        # Plot predictions for a random sample
        idx = 6
        source_mat_test = source_data_val[idx]['mat']
        target_mat_test = target_data_val[idx]['mat']
        eval_output_t = eval_output[idx]

        plot_adj_matrices(source_mat_test, 
                          target_mat_test, 
                          eval_output_t, 
                          idx, 
                          res_dir, 
                          file_name=f'eval_sample{idx}')
        

if __name__ == "__main__":
    main()