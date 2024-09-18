
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from dataset import UkraineDataset
from torchvision import transforms
import os


class UkraineDataModule(pl.LightningDataModule):
    def __init__(self, root_path, batch_size=16, num_workers=8, mode = "train", normalize=False, 
                 return_cloud_mask = False, sentinel_type="both", indices=None, dilate_mask=False, 
                 evaluation_mode="test", bands = [1, 2, 3], file_type="npy"):
        super().__init__()
        self.root_path = root_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.normalize = normalize
        self.return_cloud_mask = return_cloud_mask
        self.sentinel_type = sentinel_type
        self.indices = indices
        self.dilate_mask = dilate_mask
        self.mode = mode
        self.evaluation_mode = evaluation_mode
        self.bands = bands
        self.file_type = file_type


    def setup(self, stage=None):
        self.train_dataset = UkraineDataset(root_path=self.root_path, mode=self.mode, normalize=self.normalize, return_cloud_mask=self.return_cloud_mask, 
                                            sentinel_type=self.sentinel_type, indices=self.indices, dilate_mask=self.dilate_mask, bands=self.bands, file_type=self.file_type)
        
        self.val_dataset = UkraineDataset(root_path=self.root_path, mode="val", normalize=self.normalize, return_cloud_mask=self.return_cloud_mask, 
                                          sentinel_type=self.sentinel_type, indices=self.indices, dilate_mask=self.dilate_mask, bands=self.bands, file_type="tif")
        
        self.test_dataset = UkraineDataset(root_path=self.root_path, mode="test_whole", normalize=self.normalize, return_cloud_mask=self.return_cloud_mask, 
                                           sentinel_type=self.sentinel_type, indices=self.indices, dilate_mask=self.dilate_mask, bands=self.bands, file_type="tif")

    def train_dataloader(self):
        print("Using train dataloader")
        return DataLoader(self.train_dataset, 
                          batch_size=self.batch_size, 
                          shuffle=True, 
                          num_workers=self.num_workers,
                          pin_memory=True,
                          prefetch_factor=2
                          )
                          
    def val_dataloader(self):
        print("Using val dataloader")
        return DataLoader(self.val_dataset, 
                          batch_size=8,
                          shuffle=False, 
                          num_workers=8)

    def test_dataloader(self):
        if self.evaluation_mode == "val":
            print("Using val dataloader")
            return DataLoader(self.val_dataset, 
                          batch_size=1,
                          shuffle=False, 
                          num_workers=4)
        else:
            print("Using test dataloader")
            return DataLoader(self.test_dataset, 
                            batch_size=1,
                            shuffle=False, 
                            num_workers=4)
        
