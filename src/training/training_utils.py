
### Standard imports
import random
import numpy as np
import json
import sys
import os
import csv
import torch
from pytorch_lightning.callbacks import Callback
from pathlib import Path


### Set base path ###
base_path = Path(os.getcwd())
while not (base_path / '.git').exists():
    base_path = base_path.parent
print('Base path: ', base_path)

### custom imports ###
sys.path.append(str(base_path / 'src/visualization'))
sys.path.append(str(base_path / 'src/models'))

from model_eval_plots import plot_loss, plot_conf_matrix



def save_model_info(model_path, config):
    """Saves the model info to a json file.

    Args:
        model_path (str): Path to the model
        config (dict): Configuration dictionary
        datasource (str): Data source, default is gfc

    Returns:
        Saves the model info to a json file
    """

    metadata = {
        "model_name": config['model_name'],
        "model_num": config['model_num'],
        "bands": config['bands'],
        "input_nbr_S2": config['input_nbr_S2'],
        "input_nbr_S1": config['input_nbr_S1'],
        "learning_rate": config['learning_rate'],
        "weight_decay": config['weight_decay'],
        "epochs": config['epochs'],
        "batch_size": config['batch_size'],
        "deforest_weight": config['deforest_weight'],
        "iou_threshold": config['iou_threshold'],
        "mode": config['mode'],
        "loss_function": config['loss_fn'],
        "dilate_mask": config['dilate_mask'],
        "sentinel_type": config['sentinel_type'], 
        "patch_size": config['patch_size'],
        "kernel_size": config['kernel_size'],
        "dropout": config['dropout'],
        "scheduler": config['scheduler'],
        "patience": config['patience'],
        "comment": config['comment'],
    }

    model_info_path = model_path.replace("fitted_models", "fitted_models_metadata") + ".json"
    with open(model_info_path, 'w') as f:
        json.dump(metadata, f)
    print(f"Model info saved to {model_info_path}")
