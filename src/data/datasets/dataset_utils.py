
### Import necessary libraries ###
import os
import csv
import re
import shutil
import glob
import random
import rasterio
from rasterio.windows import Window
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import imageio.v2 as imageio
import pandas as pd
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation
from sklearn.model_selection import train_test_split


### Set base path ###
base_path = Path(os.getcwd())
while not (base_path / '.git').exists():
    base_path = base_path.parent
print('Base path: ', base_path)





def dilate_mask_scipy(mask):
    """
    Dilates a binary mask using SciPy.
    
    Args:
        mask (numpy.ndarray): The binary mask to dilate.
    
    Returns:
        numpy.ndarray: The dilated mask.
    """
    mask_bool = mask.astype(bool)
    dilated_mask = binary_dilation(mask_bool).astype(np.uint8)  
    return dilated_mask


def location_from_path(path):
    """ Extracts the location from the path of a file.
    
    Args:
    path (str): The path of the file.
    
    Returns:
    location (str): The location extracted from the path.
    """
    
    splits = os.path.basename(path).split('_')
    tile_id = splits[4]
    position1 = splits[6]
    position2 = splits[7].split('.')[0]

    return f"{tile_id}_{position1}_{position2}"


def numerical_sort_key(filename):
    """Extracts the first number from the filename and returns it as an integer for sorting.

    Args:
    filename (str): The filename to extract the number from.

    Returns:
    int: The first number extracted from the filename.
    """

    numbers = re.findall(r'\d+', os.path.basename(filename))
    return int(numbers[0]) if numbers else 0  # Convert the first number to an integer for sorting

def list_files(directory):
    """ List filenames in a directory, sorted numerically based on the first numeric part of the filename

    Args:
    directory (str): The directory to list the filenames from.

    Returns:
    list: The list of filenames sorted numerically based on the first numeric part of the filename.
    """

    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    return sorted(files, key=numerical_sort_key)  # Apply the sorting key


def save_filenames_to_csv(root_path, output_csv_path, ):
    """ Save the filenames of the dataset to a CSV file. 
    if dataset_type is 'change', the CSV file will have columns for A, B, and Masks.
    if dataset_type is 'classification', the CSV file will have columns for Image and Masks.

    Args:
    root_path: str
    output_csv_path: str
    dataset_type: str (options: 'change', 'classification')

    Returns:
    saves a CSV file with the filenames of the dataset.
    """

    s2a_path = os.path.join(root_path, 'S2_A')
    s2b_path = os.path.join(root_path, 'S2_B')
    s1a_path = os.path.join(root_path, 'S1_A')
    s1b_path = os.path.join(root_path, 'S1_B')
    clouds_a_path = os.path.join(root_path, 'clouds_A')
    clouds_b_path = os.path.join(root_path, 'clouds_B')
    masks_path = os.path.join(root_path, 'masks')
    masks_dilated_path = os.path.join(root_path, 'masks_dilated')

    
    s2a_files = list_files(s2a_path)
    s2b_files = list_files(s2b_path)
    s1a_files = list_files(s1a_path)
    s1b_files = list_files(s1b_path)
    clouds_a_files = list_files(clouds_a_path)
    clouds_b_files = list_files(clouds_b_path)
    mask_files = list_files(masks_path)
    masks_dilated_files = list_files(masks_dilated_path)

    with open(output_csv_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['S2_A', 'S2_B', 'S1_A', 'S1_B', 'clouds_A', 'clouds_B', 'masks', 'masks_dilated'])
        
        max_len = max(len(s2a_files), len(s2b_files), len(s1a_files), len(s1b_files), len(clouds_a_files), len(clouds_b_files), len(mask_files), len(masks_dilated_files))
        for i in range(max_len):
            row = [
                    s2a_files[i] if i < len(s2a_files) else '',
                    s2b_files[i] if i < len(s2b_files) else '',
                    s1a_files[i] if i < len(s1a_files) else '',
                    s1b_files[i] if i < len(s1b_files) else '',
                    clouds_a_files[i] if i < len(clouds_a_files) else '',
                    clouds_b_files[i] if i < len(clouds_b_files) else '',
                    mask_files[i] if i < len(mask_files) else '',
                    masks_dilated_files[i] if i < len(masks_dilated_files) else ''
            ]
            writer.writerow(row)


def get_location_and_id_from_filename(filename, file_type):
    """
    Extract the location information and common identifier from the filename.
    
    Args:
    filename (str): Filename of the image or mask.
    file_type (str): Type of the file ('A', 'B', 'masks', 'masks_dilated').
    
    Returns:
    tuple: (Location identifier, common identifier)
    """
    parts = filename.split('_')
    common_id = parts[1]  # The second part of the filename is the common identifier
    if file_type in ['S2_A', 'S2_B', 'S1_A', 'S1_B', 'clouds_A', 'clouds_B']:
        location = '_'.join(parts[-2:])[:-4]
    else:
        location = '_'.join(parts[-3:-1])
    return location, int(common_id)


def contains_minority_class(mask_path, minority_threshold=1):
    """
    Check if the mask contains the minority class based on the given threshold.
    
    Args:
    mask_path (str): Path to the mask file.
    minority_threshold (int): Threshold value for the minority class presence.
    
    Returns:
    bool: True if minority class is present, False otherwise.
    """
    return int(mask_path.split('_')[-1].split('.')[0]) > minority_threshold


def select_images_and_copy(dataset_dir, target_dir, minority_threshold=1, num_without_minority=1):
    """
    Select images based on the presence of the minority class and copy them to the target directory.
    
    Args:
    dataset_dir (str): Directory containing the dataset with subfolders A, B, masks, and masks_dilated.
    target_dir (str): Directory to save the selected images and masks.
    minority_threshold (int): Threshold value for the minority class presence.
    """
    # Create target directories if they don't exist
    subfolders = ['S2_A', 'S2_B', 'S1_A', 'S1_B', 'masks', 'masks_dilated', 'clouds_A', 'clouds_B']
    for subfolder in subfolders:
        os.makedirs(os.path.join(target_dir, subfolder), exist_ok=True)
    
    # Organize images and masks by location
    location_images = defaultdict(lambda: {subfolder: {} for subfolder in subfolders})
    for subfolder in subfolders:
        for file_path in glob.glob(os.path.join(dataset_dir, subfolder, '*')):
            filename = os.path.basename(file_path)
            location, common_id = get_location_and_id_from_filename(filename, subfolder)
            location_images[location][subfolder][common_id] = file_path

    print(f"Found {len(location_images)} locations.")
    
    selected_files = {subfolder: [] for subfolder in subfolders}
    
    # Filter images based on the presence of the minority class in masks
    for location, files_dict in tqdm(location_images.items()):

        masks_with_minority = {common_id: mask_path for common_id, mask_path in files_dict['masks'].items() if contains_minority_class(mask_path, minority_threshold)}
        
        if masks_with_minority:
            for common_id in masks_with_minority.keys():
                for subfolder in subfolders:
                    selected_files[subfolder].append(files_dict[subfolder][common_id])
        elif num_without_minority > 0:
            random_ids = random.sample(list(files_dict['S2_A'].keys()), min(num_without_minority, len(files_dict['S2_A'].keys())))  # Choose random IDs
            print(random_ids)
            for random_id in random_ids:
                for subfolder in subfolders:
                    print(files_dict[subfolder][random_id])
                    selected_files[subfolder].append(files_dict[subfolder][random_id])
    
    
    # Copy selected files to the target directory
    for subfolder in tqdm(subfolders):
        for file_path in tqdm(selected_files[subfolder]):
            shutil.copy(file_path, os.path.join(target_dir, subfolder, os.path.basename(file_path)))


def divide_images(source_root, dest_root, patch_size=256):
    """
    Divide the images in the source directory into patches and save them in the destination directory.

    Args:
    source_root (str): The root directory containing the images to divide.
    dest_root (str): The root directory to save the divided images.
    patch_size (int): The size of the patches to divide the images into.

    Returns:
    saves the divided images to the destination directory.
    """

    # Create destination directory if it doesn't exist
    os.makedirs(dest_root, exist_ok=True)
    
    # Iterate over subfolders like 'train', 'val', 'test'
    for subfolder in tqdm(['train', 'val', 'test', 'train_selected']):
        subfolder_path = os.path.join(source_root, subfolder)
        dest_subfolder_path = os.path.join(dest_root, subfolder)
        
        # Create subdirectories for images A, B and labels in the new root
        for category in tqdm(['S1_A', 'S1_B', 'S2_A', 'S2_B', 'masks', 'masks_dilated', 'clouds_A', 'clouds_B']):
            src_category_path = os.path.join(subfolder_path, category)
            dest_category_path = os.path.join(dest_subfolder_path, category)
            os.makedirs(dest_category_path, exist_ok=True)

            # Process each image in the category folder
            for img_name in os.listdir(src_category_path):
                img_path = os.path.join(src_category_path, img_name)

                with rasterio.open(img_path) as src:
                    width = src.width
                    height = src.height
                    channels = src.count

                    counter = 0
                # Divide the image into patches and save each patch
                    for i in range(0, width, patch_size):
                        for j in range(0, height, patch_size):

                            counter += 1
                             
                            window = Window(i, j, patch_size, patch_size)
                            img_patch = src.read(window=window)
                                
                            # Define the window for each patch
                            if img_patch.shape[1] < patch_size or img_patch.shape[2] < patch_size:
                                continue
                            
                            splits = img_name.split('_')
                            first_part = '_'.join(splits[0:2])
                            second_part = '_'.join(splits[2:])
                            patch_name = f'{first_part}.{counter}_{second_part}'
                            patch_path = os.path.join(dest_category_path, patch_name)

                            # Save the patch as a TIFF file with the same format
                            with rasterio.open(
                                patch_path,
                                'w',
                                driver='GTiff',
                                height=img_patch.shape[1],
                                width=img_patch.shape[2],
                                count=channels,
                                dtype=img_patch.dtype,
                                crs=src.crs,
                                transform=src.window_transform(window),
                            ) as dst:
                                dst.write(img_patch)


def read_tiff(file_path):
    image = Image.open(file_path)
    return np.array(image)


def save_tiff(mask, output_path):
    image = Image.fromarray(mask.astype(np.uint8))
    image.save(output_path)


def change_labels_in_circular_area(mask, center, radius, new_label):

    """Change the labels in a circular area of the mask.

    Args:
    mask (numpy.ndarray): The mask to modify.
    center (tuple): The center of the circular area.
    radius (int): The radius of the circular area.
    new_label (int): The new label to assign to the circular area.

    Returns:
    numpy.ndarray: The modified mask.
    """

    y, x = np.ogrid[:mask.shape[0], :mask.shape[1]]
    dist_from_center = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    mask[dist_from_center <= radius] = new_label
    return mask


def process_adjustments(csv_path, input_folder, output_folder):

    """
    Process the adjustments in the masks based on the CSV file.

    Args:
    csv_path (str): The path to the CSV file containing the adjustments.
    input_folder (str): The folder containing the masks to adjust.
    output_folder (str): The folder to save the adjusted masks.

    Returns:
    saves the adjusted masks to the output folder.
    """

    df = pd.read_csv(csv_path, sep=';')
    
    os.makedirs(output_folder, exist_ok=True)

    mask_files = sorted(glob.glob(os.path.join(input_folder, "*.tif")), key=numerical_sort_key)
    print(mask_files)

    for mask_path in mask_files:
        
        mask_name = os.path.basename(mask_path)
        mask = read_tiff(mask_path)


        if int(mask_name.split('_')[1]) in list(df['Index']):
            
            for index, row in df.iterrows():

                dataset = row['Dataset']
                index = row['Index']
                adjustment = row['Adjustment']
                center1 = row['center1']
                center2 = row['center2']
                radius = row['radius']

                if dataset.lower() == mask_name.split('_')[0] and index == int(mask_name.split('_')[1]):

                    print(f"Processing {mask_name}...")

                    if adjustment == 'add':
                        center = (int(center1), int(center2))
                        radius = float(radius)
                        mask = change_labels_in_circular_area(mask, center, radius, 1)
                    elif adjustment == 'remove':
                        if center1 == '-' and center2 == '-' and radius == '-':
                            mask[:] = 0
                        else:
                            center = (int(center1), int(center2))
                            radius = int(radius)
                            mask = change_labels_in_circular_area(mask, center, radius, 0)

            # change the deforesation number in the mask file
            split = os.path.basename(mask_path).split('_')
            new_deforestation_number = mask.sum()
            split[-1] = f"{new_deforestation_number}.tif"
            new_mask_name = '_'.join(split)

    
            # Save the modified mask
            output_path = os.path.join(output_folder, new_mask_name)
            save_tiff(mask, output_path)

            print(f"Processed {mask_name} successfully.")
        
        else:
            # change the deforesation number in the mask file
            split = os.path.basename(mask_path).split('_')
            new_deforestation_number = mask.sum()
            split[-1] = f"{new_deforestation_number}.tif"
            new_mask_name = '_'.join(split)

            output_path = os.path.join(output_folder, new_mask_name)
            shutil.copy(mask_path, output_path)

    

def sample_train_data(file_dir, output_folder, sample_size=0.5, random_state=42):
    """
    Sample a subset of the training data based on the deforestation class distribution.

    Args:
    file_dir (str): The directory containing the training data.
    output_folder (str): The directory to save the sampled data.
    sample_size (float): The fraction of the data to sample.
    random_state (int): The random seed for reproducibility.
    
    Returns:
    saves the sampled data to the output folder.
    """

    file_path = os.path.join(file_dir, 'masks_dilated')
    files = os.listdir(file_path)

    data = []
    for file in files:
        parts = file.split('_')
        file_id = parts[1]
        deforestation = int(parts[-1].split('.')[0])
        data.append({'ID': file_id, 'Deforestation': deforestation})

    df = pd.DataFrame(data)

    print(f"Total {len(df)} images.")

    value_counts = df['Deforestation'].value_counts()
    to_keep = value_counts[value_counts > 1].index  # Get indices of classes with more than one sample
    df_filtered = df[df['Deforestation'].isin(to_keep)]

    print(f"Total {len(df_filtered)} images after filtering.")

    # If no categories are left, raise an error or handle the case
    if df_filtered.empty:
        raise ValueError("All categories have too few samples for stratification after filtering. Adjust sample_size or data collection strategy.")

    # Stratified split
    train, sample = train_test_split(df_filtered, test_size=sample_size, random_state=random_state, stratify=df_filtered['Deforestation'])


    print(f"Sampled {len(sample)} images.")
    print(f"Sampled {len(train)} images.")

    # copy files based on id
    for dir in tqdm(['S2_A', 'S2_B', 'S1_A', 'S1_B', 'masks', 'masks_dilated', 'clouds_A', 'clouds_B']):
        file_path = os.path.join(file_dir, dir)
        output_path = os.path.join(output_folder, dir)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        for index, row in sample.iterrows():
            file_id = row['ID']
            files = glob.glob(os.path.join(file_path, f'*_{file_id}_*'))
            for file in files:
                shutil.copy(file, output_path)


### Usage for sampling training data ###
if __name__ == 'sample_train_data':
    file_dir = str(base_path / 'data/UKR/final_datasets/change_new/train')
    output_folder = str(base_path / 'data/UKR/final_datasets/change_new/train_sampled')
    sample_train_data(file_dir, output_folder, sample_size=0.1, random_state=42)

### Usage for labeling adjustments ###
if __name__ == 'label_adjustments':
    csv_path = str(base_path / 'data/UKR/final_datasets/change_new/Label_adjustments.csv')
    input_folder = str(base_path / 'data/UKR/final_datasets/change_new/train/masks_dilated')
    output_folder = str(base_path / 'data/UKR/final_datasets/change_new/train/masks_dilated_adjusted')

    process_adjustments(csv_path, input_folder, output_folder)

### Usage for saving filenames to CSV ###
if __name__ == 'filename_to_csv':
    root_path = str(base_path / 'data/UKR/final_datasets/change_new/val')
    output_csv_path = str(base_path / 'data/UKR/final_datasets/change_new/val/filenames_val.csv')

    save_filenames_to_csv(root_path, output_csv_path)

### Usage for dividing images into patches ###
if __name__ == 'divide_images':
    source_root = str(base_path / 'data/UKR/final_datasets/change_new/train')
    dest_root = str(base_path / 'data/UKR/final_datasets/change_new/train_patches')
    divide_images(source_root, dest_root, patch_size=256)

### Usage for selecting images with minority class ###
if __name__ == 'select_images':
    dataset_dir = str(base_path / 'data/UKR/final_datasets/change_new/train')
    target_dir = str(base_path / 'data/UKR/final_datasets/change_new/train_selected')
    minority_threshold = 20  # Adjust threshold as needed
    num_without_minority = 3 # Number of images to select without the minority class
    select_images_and_copy(dataset_dir, target_dir, minority_threshold, num_without_minority)

