import os
import sys
import csv
import re
import imageio.v2 as imageio
import rasterio
import numpy as np
import pandas as pd

from tqdm import tqdm
from geopandas import GeoSeries
from shapely.geometry import Polygon
from rasterio.windows import Window
from rasterio.plot import reshape_as_image



def calculate_index(band_a, band_b, scale = 0.0001):
    """Calculates the index of two bands. 
    NDVI = (NIR - Red) / (NIR + Red)
    NDMI = (8a - SWIR1) / (8a + SWIR1)
    """
    band_a = band_a.astype(np.float32) * scale
    band_b = band_b.astype(np.float32) * scale

    with np.errstate(divide='ignore', invalid='ignore'):
        index = (band_b - band_a) / (band_b + band_a + 1e-9)
        index = np.nan_to_num(index)  # Replace NaN with 0
        normalized_index = (index + 1) / 2
    return normalized_index


def divide_into_pieces(image_path, save_path, width = 256, height = 256, ndvi_ndmi = False):
    """Divides satellite image into pieces of specified size. Adds NDVI and NDMI as additional bands.

    Args:
        image_path: str - path to the source image
        save_path: str - path to the directory where pieces will be stored
        width: int - width of a piece
        height: int - height of a piece
        ndvi_ndmi: bool - add NDVI and NDMI as additional bands

    Returns:
        saves tifs and geojsons in the specified directory
        writes the metadata to a csv file.
    """

    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
        print('Data directory created.')

    os.makedirs(f'{save_path}/s2', exist_ok=True)
    os.makedirs(f'{save_path}/geojson_polygons', exist_ok=True)
    
    with rasterio.open(image_path) as src, open(f'{save_path}/image_pieces.csv', 'w') as csvFile:
        
        # Initialize csv writer for the image piece info
        writer = csv.writer(csvFile)
        writer.writerow([
            'original_image', 'piece_image', 'piece_geojson',
            'start_x', 'start_y', 'width', 'height'
        ])
        

        for j in tqdm(range(0, src.height // height)):
            for i in range(0, src.width // width):

                meta = src.meta.copy()
                
                # create image piece
                window=Window(i * width, j * height, width, height)
                data = src.read(window=window)

    
                if ndvi_ndmi:
                    ndvi = calculate_index(data[0], data[3]) # Red, NIR
                    ndmi = calculate_index(data[4], data[5]) # 8a, SWIR1

                    # add NDVI and NDMI to the raster window
                    data = np.concatenate((data, ndvi[np.newaxis, :, :], ndmi[np.newaxis, :, :]))

                # Create filenames for saving
                piece_name = f'{os.path.splitext(os.path.basename(image_path))[0]}_{j}_{i}.tif'
                piece_geojson_name = f'{os.path.splitext(os.path.basename(image_path))[0]}_{j}_{i}.geojson'

                # Update metadata
                meta.update({
                    'width': width,
                    'height': height,
                    'count': data.shape[0],
                    'dtype': 'float32' if ndvi_ndmi else src.meta['dtype']
                })

                # Save image piece
                with rasterio.open(f'{save_path}/s2/{piece_name}', 'w', **meta) as dst:
                    for band_index in range(data.shape[0]):
                        band_data = data[band_index, :, :]
                        dst.write(band_data, band_index + 1)

                # Create and save the polygon geojson
                transform = src.window_transform(window)
                polygon = Polygon([
                    transform * (0, 0), transform * (0, height),
                    transform * (width, height), transform * (width, 0),
                    transform * (0, 0)
                ])
                gdf = GeoSeries([polygon], crs=src.crs)
                gdf.to_file(f'{save_path}/geojson_polygons/{piece_geojson_name}', driver='GeoJSON')

                # Write metadata to CSV
                writer.writerow([os.path.basename(image_path), piece_name, piece_geojson_name, i * width, j * height, width, height])

    csvFile.close()


def split_tif(tif_path, save_tif_path, image_pieces_info_path):
    """Splits a TIFF image into pieces according to the image pieces info and saves them.
    
    Args:
        tif_path (str): Path to the TIFF file.
        save_tif_path (str): Path to the directory where the pieces will be saved.
        image_pieces_path (str): Path to the image pieces info file.

    Returns:
        Saves the image pieces in the save_tif_path directory.
    """

    # Ensure the save path exists
    if not os.path.exists(save_tif_path):
        os.makedirs(save_tif_path)

    # Read the image pieces info
    pieces_info = pd.read_csv(
        image_pieces_info_path, dtype={
            'start_x': int, 'start_y': int,
            'width': int, 'height': int
        }
    )

    # Open the TIFF file
    with rasterio.open(tif_path) as src:
        for i in tqdm(range(pieces_info.shape[0])):
            piece = pieces_info.iloc[i]
            window = rasterio.windows.Window(
                piece['start_x'],
                piece['start_y'],
                piece['width'],
                piece['height']
            )

            # Read the data from the window
            data = src.read(window=window)

            # Define the transformation for the new piece
            transform = src.window_transform(window)

            # Create a new raster file for the piece
            output_path = os.path.join(save_tif_path, f"{re.split(r'[/.]', piece['piece_image'])[-2]}.tif".replace('S2A', 'S1'))
            with rasterio.open(
                output_path, 'w',
                driver='GTiff',
                height=piece['height'],
                width=piece['width'],
                count=src.count,
                dtype=src.dtypes[0],
                crs=src.crs,
                transform=transform
            ) as dst:
                dst.write(data)


