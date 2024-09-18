import os
import glob
import cv2
import rasterio
import random
import time
from PIL import Image
from functools import lru_cache
import numpy as np
import torch
from torch.utils.data import Dataset
import random
from dataset_utils import numerical_sort_key


    
class Normalize:
    """Scale image tensor to [0, 1]."""
    def __init__(self, MAX, MIN):
        self.MAX = MAX
        self.MIN = MIN
    def __call__(self, image):
        for i in range(image.shape[0]):
            image[i] = torch.clamp((image[i] - self.MIN[i]) / (self.MAX[i] - self.MIN[i]), 0, 1)
        return image


MEAN_S2 = [877.2998022165781, 1043.3608926402433, 920.190422762149, 3104.212542326554, 3224.922812921343, 2364.990418726084, 1608.7049464147071] # R, G, B, NIR, B8a, SWIR1, SWIR2,
STD_S2 = [687.1702898205211, 683.3690635189156, 763.2896601166996, 1059.2718616955108, 1059.984087409077, 879.2087245418816, 899.0865518272087]
MIN_S2 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] # R, G, B, NIR, B8a, SWIR1, SWIR2,
MAX_S2 = [19344.0, 19472.0, 21344.0, 16800.0, 16504.0, 16196.0, 16119.0]

MEAN_S1 = [-10.934213151755788, -10.934213151755788]
STD_S1 = [3.3365800231293985, 3.3365800231293985]
MIN_S1 = [-72.49531432193484, -72.49531432193484]
MAX_S1 = [23.442128030399232, 23.442128030399232]


class UkraineDataset(Dataset):
    def __init__(self, root_path=None, mode = "train", normalize = False, return_cloud_mask = False, 
                 sentinel_type = "both", indices = None, dilate_mask = False, bands = [1, 2, 3], file_type = "npy"):
    
        """
        Args:
            root_path (str): Path to the dataset folder.
            mode (str): Dataset mode, ['train', 'train_selected', 'train_augmented', 'val', 'test']
            normalize (bool): Normalize the image bands.
            return_cloud_mask (bool): Return cloud mask.
            sentinel_type (str): Sentinel type, ['S2', 'S1', 'both']
            indices (list): List of indices to select specific images.
            dilate_mask (bool): Dilate the mask.
            bands (list): List of bands to load.
            file_type (str): File type, ['tif', 'npy']
        """

        folder_path = os.path.join(root_path, mode)
        self.dilate_mask = "masks_dilated" if dilate_mask else "masks"
        self.file_type = file_type
        self.sentinel_type = sentinel_type
        self.normalize = normalize
        self.return_cloud_mask = return_cloud_mask
        if file_type == "tif":
            self.bands = bands
        elif file_type == "npy":
            self.bands = [band - 1 for band in bands]
        else:
            raise ValueError("Invalid file type. Choose between 'tif' and 'npy'.")

        if sentinel_type in ["S2", "S1"]:
            print(os.path.join(folder_path, f"{sentinel_type}_A/*.{file_type}"))

            self.A_files = sorted(glob.glob(os.path.join(folder_path, f"{sentinel_type}_A/*.{file_type}")), key=numerical_sort_key)
            self.B_files = sorted(glob.glob(os.path.join(folder_path, f"{sentinel_type}_B/*.{file_type}")), key=numerical_sort_key)
            self.mask_files = sorted(glob.glob(os.path.join(folder_path, f"{self.dilate_mask}/*.{file_type}")), key=numerical_sort_key)

            print(f"Number of images: {len(self.A_files)}")

        elif sentinel_type == "both":
            self.S2_A_files = sorted(glob.glob(os.path.join(folder_path, f"S2_A/*.{file_type}")), key=numerical_sort_key)
            self.S2_B_files = sorted(glob.glob(os.path.join(folder_path, f"S2_B/*.{file_type}")), key=numerical_sort_key)
            self.S1_A_files = sorted(glob.glob(os.path.join(folder_path, f"S1_A/*.{file_type}")), key=numerical_sort_key)
            self.S1_B_files = sorted(glob.glob(os.path.join(folder_path, f"S1_B/*.{file_type}")), key=numerical_sort_key)
            self.mask_files = sorted(glob.glob(os.path.join(folder_path, f"{self.dilate_mask}/*.{file_type}")), key=numerical_sort_key)

            if indices is not None:
                self.S2_A_files = [self.S2_A_files[i] for i in indices]
                self.S2_B_files = [self.S2_B_files[i] for i in indices]
                self.S1_A_files = [self.S1_A_files[i] for i in indices]
                self.S1_B_files = [self.S1_B_files[i] for i in indices]
                self.mask_files = [self.mask_files[i] for i in indices]

        else:
            raise ValueError("Invalid sentinel type. Choose between 'S2', 'S1', 'both'.")
        
        # load cloud mask
        if self.return_cloud_mask:
            self.cloud_mask_A = sorted(glob.glob(os.path.join(folder_path, f"clouds_A/*.{file_type}")), key=numerical_sort_key)
            self.cloud_mask_B = sorted(glob.glob(os.path.join(folder_path, f"clouds_B/*.{file_type}")), key=numerical_sort_key)


    def __getitem__(self, index):

        if self.sentinel_type == "both":
            patchA = self.load_image(self.S2_A_files[index], sentinel_type = "S2")
            patchB = self.load_image(self.S2_B_files[index], sentinel_type = "S2")
            patchA2 = self.load_image(self.S1_A_files[index], sentinel_type = "S1")
            patchB2 = self.load_image(self.S1_B_files[index], sentinel_type= "S1")
        else:
            patchA = self.load_image(self.A_files[index], sentinel_type = self.sentinel_type)
            patchB = self.load_image(self.B_files[index], sentinel_type = self.sentinel_type)
            patchA2 = None
            patchB2 = None

        mask = self.load_mask(self.mask_files[index])


        patchA = torch.tensor(patchA, dtype=torch.float32)
        patchB = torch.tensor(patchB, dtype=torch.float32)
        patchA2 = torch.tensor(patchA2, dtype=torch.float32) if patchA2 is not None else None
        patchB2 = torch.tensor(patchB2, dtype=torch.float32) if patchB2 is not None else None
        mask = torch.tensor(mask, dtype=torch.long)

        if self.normalize:
            normalize_S2 = Normalize(MAX_S2, MIN_S2)
            normalize_S1 = Normalize(MAX_S1, MIN_S1)
            patchA = normalize_S1(patchA) if self.sentinel_type == "S1" else normalize_S2(patchA)
            patchB = normalize_S1(patchB) if self.sentinel_type == "S1" else normalize_S2(patchB)
            patchA2 = normalize_S1(patchA2) if patchA2 is not None else None
            patchB2 = normalize_S1(patchB2) if patchB2 is not None else None

        
        if self.return_cloud_mask:
            clouds_A = self.load_mask(self.cloud_mask_A[index])
            clouds_B = self.load_mask(self.cloud_mask_B[index])

            clouds_A = torch.tensor(clouds_A).long()
            clouds_B = torch.tensor(clouds_B).long()


        if self.return_cloud_mask:
            return {
                "A": patchA,
                "B": patchB,
                "A2": patchA2,
                "B2": patchB2,
                "mask": mask.long(),
                "path": [self.S2_A_files[index], self.S2_B_files[index], self.S1_A_files[index], self.S1_B_files[index], self.mask_files[index], self.cloud_mask_A[index], self.cloud_mask_B[index]],
                "cloud_mask_A": clouds_A,
                "cloud_mask_B": clouds_B
            } if self.sentinel_type == "both" else {
                "A": patchA,
                "B": patchB,
                "mask": mask.long(),
                "path": [self.A_files[index], self.B_files[index], self.mask_files[index], self.cloud_mask_A[index], self.cloud_mask_B[index]],
                "cloud_mask_A": clouds_A,
                "cloud_mask_B": clouds_B
            }
        else: 
            return {
                "A": patchA,
                "B": patchB,
                "A2": patchA2,
                "B2": patchB2,
                "mask": mask.long(),
                "path": [self.S2_A_files[index], self.S2_B_files[index], self.S1_A_files[index], self.S1_B_files[index], self.mask_files[index], ]
            } if self.sentinel_type == "both" else {
                "A": patchA,
                "B": patchB,
                "mask": mask.long(),
                "path": [self.A_files[index], self.B_files[index], self.mask_files[index]]
            }

    def __len__(self):
        if self.sentinel_type == "both":
            return len(self.S2_A_files)
        else:
            return len(self.A_files)
    
    def load_image(self, path, sentinel_type = "S2"):
        if self.file_type == "tif":
            if sentinel_type == "S2":
                return rasterio.open(path).read(self.bands).astype(np.float32)
            elif sentinel_type == "S1":
                return rasterio.open(path).read().astype(np.float32)
        elif self.file_type == "npy":
            if sentinel_type == "S2":
                return np.load(path)[self.bands, :, :].astype(np.float32)
            elif sentinel_type == "S1":
                return np.load(path).astype(np.float32)
        else:
            raise ValueError("Invalid file type. Choose between 'tif' and 'npy'.")
        
    def load_mask(self, path):
        if self.file_type == "tif":
            return cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        elif self.file_type == "npy":
            return np.load(path)

    