

### standard imports
import sys
from pathlib import Path
import csv
import os
import yaml
import json
import numpy as np

### pytorch imports
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

### Set base path ###
base_path = Path(os.getcwd())
while not (base_path / '.git').exists():
    base_path = base_path.parent
print('Base path: ', base_path)


### Import custom functions ###
sys.path.append(str(base_path / 'src/data/datasets'))
sys.path.append(str(base_path / 'src/evaluation'))
sys.path.append(str(base_path / 'src/visualization'))
sys.path.append(str(base_path / 'src/models'))

from data_modules import UkraineDataModule
from cd_model import CDModel
from training_utils import save_model_info


def main(config):

    # Define the model path
    model_save_name = f"{config['model_name']}_ukr_cd_{config['model_num']}"
    model_path = os.path.join(str(base_path / 'models/fitted_models/UKR_cd'), model_save_name)

    # Initialize the DataModule
    data_module = UkraineDataModule(root_path=config['root_path'], batch_size=config['batch_size'], num_workers=4, mode = config['mode'],
                                    normalize=True, sentinel_type=config["sentinel_type"], indices=None, dilate_mask=config['dilate_mask'],
                                    bands=config['bands'], file_type=config['file_type'])

    
    # Initialize the model
    model = CDModel(model = config['model_name'],
                    in_channels_S2 = config['input_nbr_S2'],
                    in_channels_S1 = config['input_nbr_S1'],
                    num_classes = config['num_classes'],
                    deforest_weight = config['deforest_weight'],
                    loss = config['loss_fn'],
                    lr = config['learning_rate'],
                    weight_decay = config['weight_decay'],
                    patience = config['patience'],
                    scheduler = config['scheduler'],
                    sentinel_type = config['sentinel_type'],
                    kernel_size = config['kernel_size'],
                    dropout = config['dropout'],
                    iou_threshold = config['iou_threshold'],)

    # Initialize the trainer
    seed_everything(config['seed'], workers=True)
    checkpoint_callback = ModelCheckpoint(monitor='val_loss', save_top_k=1, mode='min', filename=model_save_name,
                                          dirpath = str(base_path / 'models/fitted_models/UKR_cd'))
    
    csv_logger = CSVLogger(save_dir=str(base_path / 'models/logs/UKR_cd'), name=model_save_name)
    trainer = Trainer(max_epochs=config['epochs'], accelerator = 'gpu', callbacks=[checkpoint_callback], devices = 1, logger=csv_logger, enable_progress_bar=True)

    # Train the model
    trainer.fit(model, datamodule=data_module)

    # Save the model info
    save_model_info(model_path, config)


if __name__ == "__main__":
    # Load config settings
    config_path = os.path.join(base_path, 'models/configs')
    config_file = 'config_ukr_cd.yaml'
    with open(os.path.join(config_path, config_file), 'r') as file:
        config = yaml.safe_load(file)

    main(config)
