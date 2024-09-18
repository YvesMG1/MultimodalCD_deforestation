import os
import re
import datetime
import argparse
import numpy as np
import pandas as pd
import rasterio as rs
import geopandas as gp
import imageio.v2 as imageio

from rasterio import features
from tqdm import tqdm



def poly2mask(polys_paths, image_path, save_path, type_filter=None, filter_by_date=True, buffer=False):
    """Creates a mask from deforestation polygons in multiple files and saves it in the specified directory.
    
    Args:
        polys_paths (list): List of paths to the polygon files
        image_path (str): Path to the image
        save_path (str): Path to the directory where the mask will be saved
        type_filter (str): Type of clearcut: "open" or "overgrown"
        filter_by_date (bool): Filter by date is enabled
        buffer (bool): Buffer the polygons

    Returns:
        str: Path to the saved mask
    """
    
    if not os.path.exists(save_path):
        os.mkdir(save_path)
        print("Output directory created.")
    
    # Initialize a list to store GeoDataFrames from all files
    gdfs = []

    # Iterate over all polygon files and append to gdfs list
    for path in polys_paths:
        markup = gp.read_file(path)
        gdfs.append(markup)

    # Concatenate all GeoDataFrames into one GeoDataFrame
    all_polys = pd.concat(gdfs, ignore_index=True)


    ## Filter by date
    if filter_by_date:
        # Convert the date to datetime in the markup
        all_polys['img_date'] = all_polys['img_date'].apply(
            lambda x: datetime.datetime.strptime(x, '%Y-%m-%d')
        )
        
        # Get the date of the image
        original_image_filename = os.path.basename(os.path.normpath(image_path))
        dt = original_image_filename.split('_')[3][0:8]

        if dt.endswith('T'):
            dt = datetime.datetime.strptime(dt[:-1], '%Y%m%d')
        else:
            dt = datetime.datetime.strptime(dt, '%Y%m%d')
        
        # Adjust the filter to consider all polygons prior to the day after the image date
        dt += datetime.timedelta(days=1)
        polys = all_polys[all_polys['img_date'] <= dt].loc[:, 'geometry']
    else:
        polys = all_polys.loc[:, 'geometry']
    
    print('Number of polygons:', len(polys))

    # Create a mask from the polygons
    with rs.open(image_path) as image:
        polys = polys.to_crs({'init': image.crs})

        if buffer:
            xres, yres = image.res
            buffer_distance = max(xres, yres)
            polys = polys.buffer(buffer_distance)

        if type_filter in ['open', 'overgrown']:
            polys = polys[all_polys['state'] == type_filter]
        elif type_filter is not None:
            raise Exception(f'{type_filter} is unknown type! "open" or "overgrown" is available.')

        mask = features.rasterize(
            shapes=polys,
            out_shape=(image.height, image.width),
            transform=image.transform,
            fill=0,
            default_value=1,
            dtype=np.uint8
        )
    
    # Save the mask
    filename = '{}/{}.png'.format(
        save_path,
        re.split(r'[/.]', image_path)[-2]
    )
    imageio.imwrite(filename, mask)

    return filename


def split_mask(mask_path, save_mask_path, image_pieces_path):
    """Splits deforested mask into pieces according to the image pieces info and saves them.
    
    Args:
        mask_path (str): Path to the mask
        save_mask_path (str): Path to the directory where the pieces will be saved
        image_pieces_path (str): Path to the image pieces info file

    Returns:
        Saves the mask pieces in the save_mask_path directory
    """

    # Check if the save path exists
    if not os.path.exists(save_mask_path):
        os.mkdir(save_mask_path)

    # Read the image pieces info
    pieces_info = pd.read_csv(
        image_pieces_path, dtype={
            'start_x': np.int64, 'start_y': np.int64,
            'width': np.int64, 'height': np.int64
        }
    )
    # Read the full mask 
    mask = imageio.imread(mask_path)

    # Split the mask into pieces
    for i in tqdm(range(pieces_info.shape[0])):
        piece = pieces_info.loc[i]
        piece_mask = mask[
             piece['start_y']: piece['start_y'] + piece['height'],
             piece['start_x']: piece['start_x'] + piece['width']
        ]
        filename_mask = '{}/{}.png'.format(
            save_mask_path,
            re.split(r'[/.]', piece['piece_image'])[-2]
        )
        imageio.imwrite(filename_mask, piece_mask)


"""
TO BE UPDATED
"""

def parse_args():
    parser = argparse.ArgumentParser(
        description='Script for creating binary mask from geojson.')
    parser.add_argument(
        '--polys_path', '-pp', dest='polys_path',
        required=True, help='Path to the polygons'
    )
    parser.add_argument(
        '--image_path', '-ip', dest='image_path',
        required=True, help='Path to source image'
    )
    parser.add_argument(
        '--save_path', '-sp', dest='save_path',
        default='../output',
        help='Path to directory where mask will be stored'
    )
    parser.add_argument(
        '--mask_path', '-mp', dest='mask_path',
        help='Path to the mask',
        required=False
    )
    parser.add_argument(
        '--pieces_path', '-pcp', dest='pieces_path',
        help='Path to directory where pieces will be stored',
        default='../output/masks'
    )
    parser.add_argument(
        '--pieces_info', '-pci', dest='pieces_info',
        help='Path to the image pieces info'
    )
    parser.add_argument(
        '--type_filter', '-tf', dest='type_filter',
        help='Type of clearcut: "open" or "closed")'
    )
    parser.add_argument(
        '--filter_by_date', '-fd', dest='filter_by_date',
        action='store_true', default=False,
        help='Filter by date is enabled'
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    mask_path = poly2mask(
        args.polys_path, args.image_path,
        args.save_path, args.type_filter
    )
    if args.mask_path is not None:
        mask_path = args.mask_path

    split_mask(mask_path, args.pieces_path, args.pieces_info)
