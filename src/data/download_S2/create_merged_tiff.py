
import os
import subprocess
import tempfile
import sys

from utils import search_band, to_tiff

def merge_bands(output_file, *input_files):
    """Merge multiple single-band TIFFs into a single multi-band TIFF."""

    cmd = ['gdal_merge.py', '-separate', '-o', output_file, '-of', 'GTiff'] + list(input_files)
    subprocess.call(cmd)

    for file in input_files:
        os.remove(file)

def merged_tiff(data_folder, save_path, bands_10m, bands_20m):
    """Merge multiple bands from the granule and save them as a single TIFF.

    Args:
        data_folder (str): Path to the folder containing the granule
        save_path (str): Path to the directory where the merged TIFF will be saved
        bands_10m (list): List of 10m bands to include in the merged TIFF
        bands_20m (list): List of 20m bands to include in the merged TIFF

    Returns:
        Saves the merged TIFF in the save_path directory
    """


    granule_folder = os.path.join(data_folder, 'GRANULE')
    print(granule_folder)
    tile_folder = next(os.walk(granule_folder))[1][0]
    print(tile_folder)
    img_folder_10m = os.path.join(granule_folder, tile_folder, 'IMG_DATA', 'R10m')
    img_folder_20m = os.path.join(granule_folder, tile_folder, 'IMG_DATA', 'R20m')

    # Using temporary directory to handle individual band files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_files = []

        # Process 10m bands
        for band in bands_10m:
            band_file = search_band(band, img_folder_10m)
            if band_file:
                temp_tiff = os.path.join(temp_dir, f'{band}.tif')
                to_tiff(os.path.join(img_folder_10m, band_file), temp_tiff)
                temp_files.append(temp_tiff)

        # Process 20m bands, resample to 10m
        for band in bands_20m:
            band_file = search_band(band, img_folder_20m)
            if band_file:
                temp_tiff = os.path.join(temp_dir, f'{band}.tif')
                to_tiff(os.path.join(img_folder_20m, band_file), temp_tiff, resample_to=10)
                temp_files.append(temp_tiff)

        # Merge all bands into a single multi-band TIFF
        id = tile_folder.split('_')[1][1:]
        date = tile_folder.split('_')[3][:8]
        save_name = f'S2A_Image_{id}_{date}.tif'
        merged_file = os.path.join(save_path, save_name)
        merge_bands(merged_file, *temp_files)

        print(f'Merged TIFF saved at {merged_file}')


def slc_tiff(data_folder, save_path, resample_to=10):
    """Process SCL band from the granule and save it as a TIFF.

    Args:
        data_folder (str): Path to the folder containing the granule
        save_path (str): Path to the directory where the SCL TIFF will be saved
        resample_to (int): Resolution to resample the SCL band to, default is 10

    Returns:  
        Saves the SCL TIFF in the save_path directory
    """

    granule_folder = os.path.join(data_folder, 'GRANULE')
    tile_folder = next(os.walk(granule_folder))[1][0]
    img_folder_20m = os.path.join(granule_folder, tile_folder, 'IMG_DATA', 'R20m')
    
    # Find and process the SCL band
    scl_file = search_band('SCL', img_folder_20m)
    if scl_file:
        input_path = os.path.join(img_folder_20m, scl_file)
        id = tile_folder.split('_')[1][1:]
        date = tile_folder.split('_')[3][:8]
        save_name = f'S2A_Cloud_{id}_{date}.tif'
        output_path = os.path.join(save_path, save_name)
        
        to_tiff(input_path, output_path, resample_to=resample_to)
        print(f'SCL TIFF saved at {output_path}')
    else:
        print("SCL band file not found.")