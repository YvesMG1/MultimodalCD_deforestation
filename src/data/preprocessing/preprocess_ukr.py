import os
import sys
from tqdm import tqdm
import argparse
import rasterio
import numpy as np

from os.path import join, basename

# import custom functions
from split_images_pieces import divide_into_pieces, split_tif
from split_clouds_pieces import split_cloud
from polgyons_to_mask import poly2mask, split_mask
from utils import get_files


def preprocess(
    s2_path,
    s1_path,
    save_path, 
    cloud_path, 
    polys_path, 
    width = 256, 
    height = 256,
    filter_by_date = True, 
    buffer = True,
    ndvi_ndmi = False
):
    
    """Preprocesses the data:
    - divides sentinel tiff files into patches
    - creates masks from polygons and divides them into patches
    - divides cloud masks into patches

    Args:
    s2_path: str - path to the directory with sentinel 2 tiff files
    s1_path: str - path to the directory with sentinel 1 tiff files
    save_path: str - path to the directory where the data will be stored
    cloud_path: str - path to the directory with cloud masks
    polys_path: str - path to the directory with polygons
    width: int - width of a piece
    height: int - height of a piece
    filter_by_date: bool - filter by date is enabled
    buffer: bool - buffer the polygons
    ndvi_ndmi: bool - add NDVI and NDMI as additional bands

    Returns:
    Saves the data in the save_path directory
    """
    
    print(f'filter_by_date:{filter_by_date}')
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
        print("Save directory created.")
    
    for tiff_name in tqdm(get_files(s2_path)):

        ### Images
        s2_file = os.path.join(s2_path, tiff_name)

        # create pieces directory
        data_path = os.path.join(save_path, basename(s2_file[:-4]))
        if not os.path.exists(data_path):
            os.makedirs(data_path, exist_ok=True)
            print('Data directory created.')

        # divide S2 tiff file into patches
        divide_into_pieces(s2_file, data_path, width, height, ndvi_ndmi=ndvi_ndmi)

        # dividde S1 tiff file into patches
        s1_file = os.path.join(s1_path, tiff_name.replace('S2A', 'S1'))
        save_s1_path = os.path.join(data_path, 's1')
        os.makedirs(save_s1_path, exist_ok=True)
        pieces_info = os.path.join(data_path, 'image_pieces.csv')
        split_tif(s1_file, save_s1_path, pieces_info)

        ### Masks
        # Create masks
        masks_save_path = s2_path.replace('sentinel2', 'masks')
        mask_path = poly2mask(
            polys_path, s2_file, masks_save_path, filter_by_date=filter_by_date, buffer=buffer
        )

        # split masks into patches
        save_mask_pieces_path = os.path.join(data_path, 'masks')
        split_mask(mask_path, save_mask_pieces_path, pieces_info)

        ### Clouds
        cloud_file_path = os.path.join(cloud_path, tiff_name.replace('Image', 'Cloud'))
        save_cloud_path = os.path.join(data_path, 'clouds')

        # split clouds into patches
        split_cloud(cloud_file_path, save_cloud_path, pieces_info, save_mask_pieces_path)

