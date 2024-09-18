
import numpy as np
from scipy.ndimage import label

def ndvi(image):
    # Extract the red and near-infrared bands
    red = image[0, ...]
    nir = image[3, ...]
    
    # Compute the NDVI
    ndvi = (nir - red) / (nir + red)
    
    return ndvi

def nbr(image):

    # Extract the near-infrared and shortwave infrared bands
    nir = image[4, ...]
    swir = image[6, ...]
    
    # Compute the NBR
    nbr = (nir - swir) / (nir + swir)
    
    return nbr


def spatial_correction(change_map, radius, H = 224, W = 224):

    # Initialize the corrected label map
    corrected_map = np.zeros_like(change_map, dtype=bool)
    
    # Define the structuring element for the neighborhood
    structure = np.ones((2 * radius + 1, 2 * radius + 1), dtype=bool)
    
    # Label the connected components in the change map
    labeled_map, num_features = label(change_map)
    
    for i in range(1, num_features + 1):
        # Get the coordinates of the current region
        region_coords = np.argwhere(labeled_map == i)
        
        for coord in region_coords:
            u, v = coord
            # Extract the neighborhood
            neighborhood = change_map[max(0, u-radius):min(H, u+radius+1), max(0, v-radius):min(W, v+radius+1)]
            
            # Count the number of changed and unchanged pixels
            num_changed = np.sum(neighborhood)
            num_unchanged = neighborhood.size - num_changed
            
            # Apply the correction rule
            if num_changed >= num_unchanged:
                corrected_map[u, v] = True
                
    return corrected_map

