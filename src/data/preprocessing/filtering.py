import os
import shutil
import imageio
import rasterio
import argparse
import numpy as np
import pandas as pd

from tqdm import tqdm
from datetime import datetime


# import custom functions
from utils import get_folders, get_files, parse_filename, load_image, load_mask, save_mask, sort_files


def filter_poly(
    pieces_path, 
    save_path,
    num_deforestation_threshold,
    cloud_threshold,
):
    """Filter the pieces according to the criteria and save them in the new directory.
    Criteria:
    - if the piece has less deforested area than pixel threshold (pxl_size_threshold)
    - if the piece has more cloud % than cloud threshold

    Args:
    pieces_path: str - path to the directory with the pieces
    save_path: str - path to the directory where the filtered pieces will be saved
    num_deforestation_threshold: int - threshold for the number of deforested pixels
    cloud_threshold: float - threshold for the cloud percentage

    Returns:
    Saves the filtered pieces in the save_path directory
    """

    for folder in tqdm(get_folders(pieces_path)):
            
        # Read the pieces info for the specific folder
        pieces_info = pd.read_csv(os.path.join(pieces_path, folder, 'image_pieces.csv'))

        # get the paths of the pieces for the folder
        mask_pieces_path = os.path.join(pieces_path, folder, 'masks')
        clouds_pieces_path = os.path.join(pieces_path, folder, 'clouds')
        s2_pieces_path = os.path.join(pieces_path, folder, 's2')
        s1_pieces_path = os.path.join(pieces_path, folder, 's1')
        poly_pieces_path = os.path.join(pieces_path, folder, 'geojson_polygons')

        for i in tqdm(range(len(pieces_info))):

            poly_piece_name = pieces_info['piece_geojson'][i]
            filename, _ = os.path.splitext(poly_piece_name)

            # Read the mask and count the number of deforested pixels
            mask_piece_file = os.path.join(mask_pieces_path, filename + '.png')
            mask_piece = load_mask(mask_piece_file)
            num_deforestation = np.count_nonzero(mask_piece == 1)

            # Read the cloud and count the percentage of cloud
            cloud_piece_file = os.path.join(clouds_pieces_path, filename + '.tif')
            cloud_piece = load_image(cloud_piece_file)
            cloudy = np.count_nonzero(cloud_piece) / cloud_piece.size

            # Filter the pieces
            if num_deforestation >= num_deforestation_threshold and cloudy <= cloud_threshold:
                
                # Add the piece to the new directory
                add_piece(
                    os.path.join(s2_pieces_path, filename + '.tif'),
                    os.path.join(s1_pieces_path, filename.replace('S2A', 'S1') + '.tif'),
                    mask_piece_file,
                    cloud_piece_file,
                    os.path.join(poly_pieces_path, filename + '.geojson'),
                    save_path,
                )
    
    print('Filtered images saved in:', save_path)



def create_diff_masks(image_dir, mask_dir, save_path, THRESHOLD_DEF_CHANGE = 10):

    """
    Create difference masks between patches in same positioin but different dates

    Args:
    image_dir: str - path to the directory with the patches
    mask_dir: str - path to the directory with the masks
    save_path: str - path to the directory where the difference masks will be saved
    THRESHOLD_DEF_CHANGE: int - threshold for the number of deforested pixels

    Returns:
    Saves the difference masks in the save_path directory
    """

    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    
    # get the file names of the images, masks and cloud masks
    image_files = sort_files(get_files(image_dir))
    mask_files = {f.replace('.png', '.tif'): f for f in os.listdir(mask_dir) if f.endswith('.png')}

    # Load all masks into a dictionary once, at the start
    all_masks = {f: load_mask(os.path.join(mask_dir, mask_files.get(f))) for f in image_files if mask_files.get(f)}


    for i in tqdm(range(len(image_files))):
        file = image_files[i]
        tile_id, date, position_1, position_2  = parse_filename(file, mask_type='mask')
        
        mask1 = all_masks.get(file)
       
        j = i + 1
        while j < len(image_files) and parse_filename(image_files[j])[0] == tile_id and parse_filename(image_files[j])[2] == position_1 and parse_filename(image_files[j])[3] == position_2:
            
            # get the mask of the other image 
            other_file = image_files[j]
            mask2 = all_masks.get(other_file)
            other_tile_id, other_date, other_position_1, other_position_2 = parse_filename(other_file, mask_type='mask')

            # check if mask2 is not empty and cloud percentage is below threshold
            diff_mask = np.abs(mask1.astype(int) - mask2.astype(int))
            change_count = np.sum(diff_mask > 0)

            # check if the difference between the two masks is above the threshold
            if change_count >= THRESHOLD_DEF_CHANGE:
                
                # add pieces to filtered dataset
                diff_mask_save_name = f'diff_{tile_id}_{date}_{other_date}_{position_1}_{position_2}_{change_count}.png'
                save_mask(diff_mask, save_path, diff_mask_save_name)

            j += 1

    print('Difference masks saved in:', save_path)


def add_piece(s2_file, s1_file, mask_file, cloud_file, poly_file, save_path):
    """Copy the files to the new directory.
    
    Args:
    s2_file: str - path to the s2 file
    s1_file: str - path to the s1 file
    mask_file: str - path to the mask file
    cloud_file: str - path to the cloud file
    poly_file: str - path to the polygon file
    save_path: str - path to the directory where the files will be saved
    
    Returns:
    Saves the files in the save_path directory
    """

    # create new directory paths for the filtered data
    new_s2_path = os.path.join(save_path, 's2')
    new_s1_path = os.path.join(save_path, 's1')
    new_mask_path = os.path.join(save_path, 'masks')
    new_cloud_path = os.path.join(save_path, 'clouds')
    new_poly_path = os.path.join(save_path, 'geojson_polygons')

    # create directories if they do not exist
    if not os.path.exists(new_s2_path):
        os.makedirs(new_s2_path, exist_ok=True)
    if not os.path.exists(new_s1_path):
        os.makedirs(new_s1_path, exist_ok=True)
    if not os.path.exists(new_mask_path):
        os.makedirs(new_mask_path, exist_ok=True)
    if not os.path.exists(new_cloud_path):
        os.makedirs(new_cloud_path, exist_ok=True)
    if not os.path.exists(new_poly_path):
        os.makedirs(new_poly_path, exist_ok=True)
    
    # copy the file to the filtered directory
    shutil.copy(s2_file, os.path.join(new_s2_path, os.path.basename(s2_file)))
    shutil.copy(s1_file, os.path.join(new_s1_path, os.path.basename(s1_file)))
    shutil.copy(mask_file, os.path.join(new_mask_path, os.path.basename(mask_file)))
    shutil.copy(cloud_file, os.path.join(new_cloud_path, os.path.basename(cloud_file)))
    shutil.copy(poly_file, os.path.join(new_poly_path, os.path.basename(poly_file)))
