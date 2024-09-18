import os
import re
import sys
import rasterio
import numpy as np
import pandas as pd
import imageio.v2 as imageio

from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


def is_image_info(row, masks_pieces_path):
    # Construct the file path
    filename = re.split(r'[/.]', row['piece_image'])[-2]
    filepath = '{}/{}.png'.format(masks_pieces_path, filename)
    
    # Check if the file exists
    if os.path.exists(filepath):
        return 1
    else:
        return 0


def split_cloud(cloud_path, save_cloud_pieces_path, image_pieces_info, masks_pieces_path):
    """ Splits cloud mask into pieces according to the image pieces info and saves them.

    Args:
        cloud_path (str): Path to the cloud mask
        save_cloud_pieces_path (str): Path to the directory where the pieces will be saved
        image_pieces_info (str): Path to the image pieces info file
        masks_pieces_path (str): Path to the deforested mask pieces

    Returns:
        Saves the cloud mask pieces in the save_cloud_pieces_path directory
    """

    # Read the image pieces info
    pieces_info = pd.read_csv(
        image_pieces_info, dtype={
            'start_x': np.int64, 'start_y': np.int64,
            'width': np.int64, 'height': np.int64
        }
    )

    # check if path exists
    if not os.path.exists(save_cloud_pieces_path):
        os.mkdir(save_cloud_pieces_path)
        print("Output directory created.")
    
    # Check if the image piece is an image
    pieces_info['is_image'] = pieces_info.apply(lambda row: is_image_info(row,masks_pieces_path), axis=1)
    pieces_info = pieces_info[pieces_info['is_image']==1]
    print('pieces:',pieces_info.shape[0])
    
    # Read the cloud mask
    with rasterio.open(cloud_path) as src:
        clouds = src.read(1)
        
    for i in tqdm(range(pieces_info.shape[0])):

        # Get the piece info
        piece = pieces_info.iloc[i]
        piece_cloud = clouds[
                 piece['start_y']: piece['start_y'] + piece['height'],
                 piece['start_x']: piece['start_x'] + piece['width']
        ]

        # 0 - missing, 1 - defective, 3 - Cloud shadows, 7 Unclassified, 8 - Cloud medium probability, 9 - Cloud high probability, 10 - Thin cirrus, 11 - Snow or ice
        cloud_mask = np.isin(piece_cloud, [0, 1, 3, 7, 8, 9, 10, 11]).astype(np.uint8)
        
        # Save the cloud mask piece
        filename_cloud = '{}/{}.tif'.format(
            save_cloud_pieces_path,
            re.split(r'[/.]', piece['piece_image'])[-2]
        )
        
        with rasterio.open(filename_cloud, 'w', 
                           driver='GTiff', 
                           height=cloud_mask.shape[0], 
                           width=cloud_mask.shape[1], 
                           count=1, 
                           dtype=cloud_mask.dtype, 
                           rs=src.crs, 
                           transform=src.transform) as dst:
            dst.write(cloud_mask, 1)

