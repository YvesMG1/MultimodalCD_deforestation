
import os
import numpy as np
import rasterio
from tqdm import tqdm
from scipy.stats import wasserstein_distance

from utils import load_mask

def calc_histogram(image_path, bins=256, pixel_range=(0, 65535), exclude_channels=[8, 9], mask_path=None):
    """Read an image and compute normalized histograms for each channel

    Args:
        image_path (str): Path to the image
        bins (int): Number of bins in the histogram
        pixel_range (tuple): Range of pixel values
        exclude_channels (list): List of channels to exclude
        mask_path (str): Path to the mask image
    Returns:
        list: List of histograms for each channel
    """

    with rasterio.open(image_path) as src:
        histograms = []
        for band in range(1, src.count + 1):
            if band in exclude_channels:
                continue
            data = src.read(band)
            data = np.clip(data, *pixel_range)  # Clipping data to the expected range
            # if mask_path disregard pixels that are masked
            if mask_path:
                mask = load_mask(mask_path)
                data = data[mask == 0]
            hist, _ = np.histogram(data, bins=bins, range=pixel_range, density=True)
            histograms.append(hist)
        return histograms


def compute_histograms(images, image_dir, bins=256, pixel_range=(0, 65535)):
    """Compute and store histograms for each image in the directory.

    Args:
        images (list): List of image filenames
        image_dir (str): Path to the image directory
        bins (int): Number of bins in the histogram
        pixel_range (tuple): Range of pixel values
    
    Returns:
        dict: Dictionary of histograms for each image
    """

    histograms_dict = {}
    for filename in tqdm(images):
        if filename.endswith('.tif'):
            image_path = os.path.join(image_dir, filename)
            histograms = calc_histogram(image_path, bins=bins, pixel_range=pixel_range)
            histograms_dict[filename] = histograms
    return histograms_dict


def compute_average_histograms(histograms_dict, num_channels=7):
    """Compute average histograms from a dictionary of histograms for 7 channels (excluding 8 and 9)

    Args:
        histograms_dict (dict): Dictionary of histograms for each image
    
    Returns:
        list: List of average histograms for each channel
    """

    channel_histograms = {i: [] for i in range(num_channels)}  # Adjust to 7 channels
    for histograms in tqdm(histograms_dict.values()):
        for i, hist in enumerate(histograms):
            channel_histograms[i].append(hist)
    
    average_histograms = []
    for i in range(num_channels):
        average_histograms.append(np.mean(channel_histograms[i], axis=0))
    return average_histograms


def calculate_wasserstein_all(histograms_dict, average_histograms):
    """Calculate the Wasserstein distance between histograms and average histograms

    Args:
        histograms_dict (dict): Dictionary of histograms for each image
        average_histograms (list): List of average histograms for each channel
    
    Returns:
        dict: Dictionary of Wasserstein distances for each image
    """

    distances = {}
    for filename, histograms in histograms_dict.items():
        distance = calculate_wasserstein(histograms, average_histograms)
        distances[filename] = distance
    return distances


def calculate_wasserstein(histogram1, histogram2):
    """Calculate the Wasserstein distance between two histograms

    Args:
        histogram1 (list): First histogram
        histogram2 (list): Second histogram
    
    Returns:
        float: Wasserstein distance between the two histograms
    """

    distance = sum(wasserstein_distance(hist1, hist2) for hist1, hist2 in zip(histogram1, histogram2))
    return distance
