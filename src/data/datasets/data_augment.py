
import os
from tqdm import tqdm
from pathlib import Path
import random
import numpy as np
import torch
import torchvision.transforms as T
from dataset import UkraineDataset

### Set base path ###
base_path = Path(os.getcwd())
while not (base_path / '.git').exists():
    base_path = base_path.parent
print('Base path: ', base_path)


class GaussianNoise(object):
    """
    Adds Gaussian noise to the image tensor.
    """

    def __init__(self, mean=0.0, std=0.1):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean


class AdjustBrightness:
    """
    Adjusts the brightness of an image by adding a constant factor.
    """
    def __init__(self, brightness_factor):
        self.brightness_factor = brightness_factor

    def __call__(self, img):
        return img + self.brightness_factor

class AdjustContrast:
    """
    Adjusts the contrast of an image by scaling the pixel values.
    """
    def __init__(self, contrast_factor):
        self.contrast_factor = contrast_factor

    def __call__(self, img):
        mean = torch.mean(img, dim=[1, 2], keepdim=True)
        return (img - mean) * self.contrast_factor + mean

class DataAugmentation(object):
    """
    Data augmentation class for Sentinel-2 and Sentinel-1 images for change detection.

    Args:
        local_crops_scale (tuple): Scale range for random resized crop.
        local_crops_number (int): Number of local crops to generate. Default is 3.
        global_augments (int): Number of global augmentations to generate. Default is 2.

    Returns:
        list: A list of global and local augmentations.
    """
    def __init__(self, local_crops_scale, local_crops_number = 3, global_augments = 2):
        self.local_crop = T.RandomResizedCrop(224, scale=local_crops_scale, interpolation=T.InterpolationMode.BICUBIC)
        
        self.flip = T.RandomHorizontalFlip(p=0.5)
        self.brightness = AdjustBrightness(0.4) 
        self.contrast = AdjustContrast(1.4)     
        self.gaussian_blur = T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 2.0))
        self.gaussian_noise = GaussianNoise(mean=0.0, std=0.1)
        self.normalize = T.Normalize(mean=[0.5]*7, std=[0.5]*7)  
        self.local_crops_number = local_crops_number
        self.global_augments = global_augments

    def set_seed(self, seed):
        random.seed(seed)
        torch.manual_seed(seed)

    def crop_with_seed(self, seed, *images):
        cropped_images = []
        for img in images:
            if img is not None:
                self.set_seed(seed)
                cropped_images.append(self.local_crop(img))
            else:
                cropped_images.append(None)
        return cropped_images

    def apply_transforms(self, imageA, imageB, image2A, image2B, mask, cloudsA, cloudsB, local = False):
        # Set a consistent seed for reproducibility
        seed = random.randint(0, 2**32 - 1)
        
        # Apply cropping
        if local:
            imageA, imageB, image2A, image2B, mask, cloudsA, cloudsB = self.crop_with_seed(seed, imageA, imageB, image2A, image2B, mask, cloudsA, cloudsB)
        self.set_seed(seed)

        # Apply custom brightness and contrast adjustments and Gaussian blur
        if random.random() > 0.5:
            imageA = self.brightness(imageA)
            imageB = self.brightness(imageB)
            imageA = self.contrast(imageA)
            imageB = self.contrast(imageB)
            
        if random.random() > 0.5:
            imageA = self.gaussian_blur(imageA)
            imageB = self.gaussian_blur(imageB)

        # Apply Gaussian noise
        if random.random() > 0.5:
            imageA = self.gaussian_noise(imageA)
            imageB = self.gaussian_noise(imageB)

        return imageA, imageB, image2A, image2B, mask.squeeze(0), cloudsA.squeeze(0), cloudsB.squeeze(0)

    def __call__(self, imageA, imageB, image2A, image2B, mask, cloudsA, cloudsB):
        """
        Apply data augmentation to the images and mask.
        
        Args:
            imageA (torch.Tensor): Sentinel-2 image tensor.
            imageB (torch.Tensor): Sentinel-2 image tensor.
            mask (torch.Tensor): Mask tensor.
            image2A (torch.Tensor): Sentinel-1 image tensor. Default is None.
            image2B (torch.Tensor): Sentinel-1 image tensor. Default is None.
            sentinel_type (str): Type of Sentinel data ('S2', 'S1', 'both'). Default is 'both'.

        Returns:
            list: A list of global and local augmentations.
        """

        if not (torch.is_tensor(imageA) and torch.is_tensor(imageB) and torch.is_tensor(mask)):
            raise TypeError("Images and mask must be PyTorch tensors")

        global_crops = [self.apply_transforms(imageA, imageB, image2A, image2B, mask, cloudsA, cloudsB, local=False) for _ in range(self.global_augments)]
        local_crops = [self.apply_transforms(imageA, imageB, image2A, image2B, mask, cloudsA, cloudsB, local=True) for _ in range(self.local_crops_number)]

        # concatenate the global and local crops
        augmented_images = global_crops + local_crops
        
        return augmented_images


def save_as_npy(augmented_data, file_names):
    """
    Save the augmented data as numpy files.

    Args:
        augmented_data (list): A list of augmented data.
        file_names (list): A list of file names.
    
    """

    counter = 0

    for i in range(len(augmented_data)):
        for j in range(len(file_names)):

            img_path = file_names[j]
            img_name = os.path.basename(img_path)
            splits = img_name.split('_')
            first_part = '_'.join(splits[0:2])
            second_part = '_'.join(splits[2:]).replace(".tif", "")
            new_img_name = f'{first_part}.{counter}_{second_part}'

            output_path = os.path.dirname(img_path).replace("train", "train_augmented3")

            if not os.path.exists(output_path):
                os.makedirs(output_path)

            np.save(os.path.join(output_path, f"{new_img_name}"), augmented_data[i][j].numpy())
        counter += 1


# Set the paths
ROOT_PATH = str(base_path / 'data/UKR/final_datasets/change_new')
OUTPUT_PATH = str(base_path / 'data/UKR/final_datasets/change_new/train_augmented1')
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Initialize the dataset and data augmentation
dataset = UkraineDataset(ROOT_PATH, mode="train", normalize=False, sentinel_type="both", dilate_mask=True, bands=[1,2,3,4,5,6,7], file_type="tif", return_cloud_mask = True)
augment = DataAugmentation(local_crops_scale=(0.05, 0.4), local_crops_number=1, global_augments=2)


# Apply data augmentation
for i in tqdm(range(dataset.__len__())):
    data = dataset[i]

    # Unpack the data
    imageA, imageB, s1_imageA, s1_imageB, mask, cloudsA, cloudsB = data['A'], data['B'], data['A2'], data['B2'], data['mask'], data['cloud_mask_A'], data['cloud_mask_B']
    paths = data['path']

    # Apply data augmentation
    augmented_images = augment(imageA, imageB, s1_imageA, s1_imageB, mask.unsqueeze(0), cloudsA.unsqueeze(0), cloudsB.unsqueeze(0))
    
    # Save the augmented images
    save_as_npy(augmented_images, paths)