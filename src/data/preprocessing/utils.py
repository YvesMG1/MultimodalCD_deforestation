import os
import rasterio
import imageio.v2 as imageio
import numpy as np
from datetime import datetime


def get_files(path):
    """Return a list of filenames in the given directory using os.scandir()."""
    with os.scandir(path) as entries:
        return [entry.name for entry in entries if entry.is_file()]


def get_folders(path):
    """Return a list of folder names in the given directory using os.walk()."""
    return list(os.walk(path))[0][1]


def parse_filename(filename, mask_type = 'image'):
    """Return information from the filename.

    if mask_type is 'image':
    returns tile_id, date, position_1, position_2
    if mask_type is 'diff_mask':
    returns tile_id, date_1, date_2, position_1, position_2, change_count

    Args:
    filename: str - filename to parse
    type: str - type of the file: 'image' or 'diff_mask'

    Returns:
    Tuple of information from the filename
    """
    parts = filename.split('_')
    if mask_type in ['image', 'mask']:
        tile_id = parts[2]
        date = parts[3]
        position_1 = parts[4]
        position_2 = parts[5].split('.')[0]
        return tile_id, date, position_1, position_2
    
    elif mask_type == 'diff_mask':
        tile_id = parts[1]
        date_1 = parts[2]
        date_2 = parts[3]
        position_1 = parts[4]
        position_2 = parts[5]
        change_count = parts[6].split('.')[0]

        return tile_id, date_1, date_2, position_1, position_2, change_count


def get_location(mask_path, mask_type = 'diff_mask'):
    """Return the location from the diff_mask filename

    Args:
    mask_path: str - path to the mask
    mask_type: str - type of the mask: 'image', 'mask' or 'diff_mask'

    Returns:
    str: location of the mask in the format 'tile_id_position_1_position_2'

    """

    if mask_type in ['image', 'mask']:
        tile_id, date, position_1, position_2  = parse_filename(mask_path)
    elif mask_type == 'diff_mask':
        tile_id, date_1, date_2, position_1, position_2, _ = parse_filename(mask_path, mask_type='diff_mask')
    location = f'{tile_id}_{position_1}_{position_2}'
    return location


def load_image(filename):
    """Load a tiff file using rasterio."""
    with rasterio.open(filename) as src:
        return src.read()


def load_mask(filename):
    """Load a mask using imageio."""
    return imageio.imread(filename)


def save_image(image, path):
    """Save a tiff file using rasterio."""
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    with rasterio.open(
        path, 'w',
        driver='GTiff',
        height=image.shape[1],
        width=image.shape[2],
        count=image.shape[0],
        dtype=image.dtype
    ) as dst:
        dst.write(image)


def save_mask(mask, path, filename):
    """Save a mask using imageio."""
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    mask = mask.astype(np.uint8)
    full_path = os.path.join(path, filename)
    imageio.imwrite(full_path, mask)


def season_from_date(date):
    """Return the season from the date."""
    year = date.year
    dates = [
        (datetime(year, 3, 15), datetime(year, 6, 15), 'spring'),
        (datetime(year, 6, 16), datetime(year, 9, 15), 'summer'),
        (datetime(year, 9, 16), datetime(year, 11, 30), 'autumn'),
        (datetime(year, 12, 1), datetime(year, 2, 28 if (year % 4 != 0 or year % 100 == 0) and year % 400 != 0 else 29), 'winter')
    ]
    for start, end, season in dates:
        if start <= date <= end:
            return season
    return 'winter' 


def sort_files(files, type = 'image'):
    """Sort the files by;
    - image: tile_id, position_1, position_2, date
    - diff_mask: tile_id, position_1, position_2 date_1, date_2

    Args:
    files: list - list of filenames
    type: str - type of the files: 'image' or 'diff_mask'

    Returns:
    list: sorted list of filenames by the given type 
    """
    if type == 'image':
        return sorted(files, key=lambda x: (parse_filename(x)[0], parse_filename(x)[2], parse_filename(x)[3], parse_filename(x)[1]))
    elif type == 'diff_mask':
        return sorted(files, key=lambda x: (parse_filename(x, type)[0], parse_filename(x, type)[3], parse_filename(x, type)[4], parse_filename(x, type)[1], parse_filename(x, type)[2]))
