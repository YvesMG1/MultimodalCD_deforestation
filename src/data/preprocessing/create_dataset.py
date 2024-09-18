import sys
import os
import glob
import gc
import shutil
import rasterio
from PIL import Image
import numpy as np
import pandas as pd

from tqdm import tqdm
from datetime import datetime
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans

# custom imports
from histogram_eval import calc_histogram, calculate_wasserstein
from utils import get_files, get_location, parse_filename, season_from_date, sort_files, load_mask


def split_location(file_name):
    """
    Split location into tile_id, position1, and position2

    Args:
    file_name (str): File name

    Returns:
    tile_id (str): Tile ID
    position1 (float): Position 1
    position2 (float): Position 2
    """
    
    parts = file_name.split('_')
    tile_id = parts[0]
    position1 = float(parts[1])
    position2 = float(parts[2]) 

    return tile_id, position1, position2


def prepare_data_clustering(mask_directory, mask_type = 'diff_mask'):

    """
    Prepare data for clustering by extracting features from file names and locations

    Args:
    directory (str): Directory containing the masks

    Returns:
    features (np.ndarray): Features for clustering
    file_paths (list): List of file paths
    data (list): List of tile_id, position1, and position2
    """
    file_names = get_files(mask_directory)

    # Filter file names based on deforestation_threshhold
    locations = [get_location(file_name, mask_type=mask_type) for file_name in file_names]
    locations = list(set(locations))

    # Split locations into tile_id, position1, and position2
    data = [split_location(location) for location in locations]
    tile_ids = [d[0] for d in data]
    positions = np.array([[d[1], d[2]] for d in data])
    
    # Convert tile_ids using label encoder
    encoder = LabelEncoder()
    tile_ids_encoded = encoder.fit_transform(tile_ids)
    
    # Combine positions with encoded tile_ids
    features = np.hstack((positions, tile_ids_encoded[:, None])) 
    return features, data


def run_clustering(mask_directory, nclusters = 3, mask_type = 'diff_mask', random_state = 42):
    """
    Runs Kmeans clustering on mask locations

    Args:
    mask_directory (str): Directory containing the masks
    nclusters (int): Number of clusters
    deforestation_threshhold (float): Deforestation threshhold
    mask_type (str): Type of mask (diff_mask, mask)
    random_state (int): Random state for reproducibility

    Returns:
    clusters (list): List of cluster assignments
    features (np.ndarray): Features for clustering
    data (list): List of tile_id, position1, and position2 
    """

    # Prepare data for clustering
    features, data = prepare_data_clustering(mask_directory, mask_type=mask_type)

    # Normalize features
    scaler = StandardScaler()
    features_normalized = scaler.fit_transform(features)

    # Run Kmeans clustering
    kmeans = KMeans(n_clusters=nclusters, random_state=random_state, n_init = 1, init='k-means++').fit(features_normalized)
    clusters = kmeans.fit_predict(features_normalized)

    return clusters, features, data


def cloud_percentage(cloud_mask):
    """
    Calculate the percentage of cloud cover in the cloud mask

    Args:
    cloud_mask (np.array): Cloud mask

    Returns:
    cloud_percentage (float): Percentage of cloud cover
    """
    cloud_percentage = np.sum(cloud_mask) / cloud_mask.size
    return cloud_percentage


def create_mask_info_df(mask_dir, cluster_assignments, clusters, location_data, mask_type = 'diff_mask'):
    """
    Create a dataframe containing information about the masks

    Args:
    diff_mask_dir (str): Directory containing the difference masks
    cluster_assignments (dict): Dictionary containing cluster assignments for training, validation, and testing
    clusters (list): List of clusters
    location_data (list): List of tile_id, position1, and position2
    mask_type (str): Type of mask (diff_mask, mask)
    
    Returns:
    df (pd.DataFrame): Dataframe containing mask information (location, date1, date2, season1, season2, year_type, deforestation, cloud1, cloud2, wasserstein, split)
    """

    training_locations = []
    validation_locations = []
    testing_locations = []

    for cluster, data_point in zip(clusters, location_data):
        tile_id, position1, position2 = data_point
        merge_data_point = f'{tile_id}_{int(position1)}_{int(position2)}'
        if cluster in cluster_assignments['training']:
            training_locations.append(merge_data_point)
        elif cluster in cluster_assignments['validation']:
            validation_locations.append(merge_data_point)
        else:
            testing_locations.append(merge_data_point)
    data = []
    masks = get_files(mask_dir)
    
    for mask in tqdm(masks):
        
        tile_id, date1, date2, position_1, position_2, deforestation = parse_filename(mask, mask_type=mask_type)
        season = season_from_date(datetime.strptime(date1, '%Y%m%d')) 
        season2 = season_from_date(datetime.strptime(date2, '%Y%m%d'))
        location = f'{tile_id}_{int(position_1)}_{int(position_2)}'
        year_type = 'inter_year' if date1[:4] != date2[:4] else 'intra_year'
        image1_name = f'S2A_Image_{tile_id}_{date1}_{position_1}_{position_2}.tif'
        image2_name = f'S2A_Image_{tile_id}_{date2}_{position_1}_{position_2}.tif'

        hist1 = calc_histogram(os.path.join(mask_dir.replace('diff_masks', 's2'), image1_name), mask_path = os.path.join(mask_dir, mask))
        hist2 = calc_histogram(os.path.join(mask_dir.replace('diff_masks', 's2'), image2_name), mask_path = os.path.join(mask_dir, mask))
        wasserstein_dist = calculate_wasserstein(hist1, hist2)

        cloud_perc1 = cloud_percentage(load_mask(os.path.join(mask_dir.replace('diff_masks', 'clouds'), image1_name)))
        cloud_perc2 = cloud_percentage(load_mask(os.path.join(mask_dir.replace('diff_masks', 'clouds'), image2_name)))

        data.append([location, date1, date2, season, season2, year_type, int(deforestation), cloud_perc1, cloud_perc2, wasserstein_dist, 'train' if location in training_locations else 'val' if location in validation_locations else 'test'])


    columns = ['Location', 'Date1', 'Date2', 'Season1', 'Season2', 'Year Type', 'Deforestation', 'Cloud1', 'Cloud2', 'Wasserstein', 'Split']
    df = pd.DataFrame(data, columns = columns)

    return df


def select_entries(df, split_type, train_dates, valtest_dates):
    """
    Select mask entries for final dataset based on specific criteria (categories, max_images, split_type)

    Args:
    df (pd.DataFrame): Dataframe containing mask information
    split_type (str): Type of split (train, val, test)
    train_dates (list): List of training dates
    valtest_dates (list): List of validation and test dates

    Returns:
    selected (pd.DataFrame): Selected entries
    """

    selected_df = pd.DataFrame()
  
    if split_type == 'train':
        subset = df[df['Date1'].isin(train_dates) & df['Date2'].isin(train_dates)]
    else: 
        subset = df[df['Date1'].isin(valtest_dates) & df['Date2'].isin(valtest_dates)]
    selected_df = pd.concat([selected_df, subset], ignore_index=True)
    selected_df = selected_df.drop_duplicates()

    return selected_df



def create_datasets(diff_mask_dir, cluster_assignments, clusters, data, train_dates, valtest_dates, mask_type='diff_mask'):
    """
    Create training, validation and test datasets from the difference masks, cluster assignments, categories and maximum number of images.
    
    Args:
    diff_mask_dir (str): Directory containing the difference masks.
    cluster_assignments (dict): Dictionary containing the cluster assignments.
    clusters (np.array): Array containing the cluster assignments.
    data (np.array): Array containing the features.
    train_dates (list): List of training dates.
    valtest_dates (list): List of validation and test dates.
    mask_type (str): Type of mask (diff_mask, mask)

    Returns:
    train_data (pd.DataFrame): Training dataset.
    val_data (pd.DataFrame): Validation dataset.
    test_data (pd.DataFrame): Test dataset.
    """
    
    # Create dataframe with mask information and cluster assignments
    df = create_mask_info_df(diff_mask_dir, cluster_assignments, clusters, data, mask_type=mask_type)
    print(df.head())
    print("Dataframe created.")

    test_data = df[df['Split'] == 'test']
    val_data = df[df['Split'] == 'val']
    train_data = df[df['Split'] == 'train']

    # Select entries for the final dataset
    test_data = test_data.groupby('Location').apply(select_entries, split_type='test', train_dates=train_dates, valtest_dates=valtest_dates).reset_index(drop=True)
    val_data = val_data.groupby('Location').apply(select_entries, split_type='val', train_dates=train_dates, valtest_dates=valtest_dates).reset_index(drop=True)
    train_data = train_data.groupby('Location').apply(select_entries, split_type='train', train_dates=train_dates, valtest_dates=valtest_dates).reset_index(drop=True)

    print(f"Training Data: {len(train_data)}")
    print(f"Validation Data: {len(val_data)}")
    print(f"Test Data: {len(test_data)}")

    # Images containing deforestation 
    print("Images containing deforestation:")
    print(f"Training Data: {len(train_data[train_data['Deforestation'] > 0])}")
    print(f"Validation Data: {len(val_data[val_data['Deforestation'] > 0])}")
    print(f"Test Data: {len(test_data[test_data['Deforestation'] > 0])}")

    return train_data, test_data, val_data, df

    

def copy_to_final_directory(df_info, diff_mask_dir, final_dir, split_type='train', save_format='npy'):
    """
    Copy selected images to the final directory for the dataset

    Args:
    df_info (pd.DataFrame): Dataframe containing mask information of selected images
    diff_mask_dir (str): Directory containing the difference masks
    final_dir (str): Directory to store the final dataset
    split_type (str): Type of split (train, val, test)
    save_format (str): Format to save the images (npy, tif)
    """

    required_dirs = ['S2_A', 'S2_B', 'S1_A', 'S1_B', 'masks', 'clouds_A', 'clouds_B']
    for dir_name in required_dirs:
        os.makedirs(os.path.join(final_dir, dir_name), exist_ok=True)

    images_dir = diff_mask_dir.replace('diff_masks', 's2')
    

    for idx, row in tqdm(df_info.iterrows(), total=len(df_info)):

        # get the data from the row
        location, *rest = row
        tile_id, position_1, position_2 = location.split('_')

        def copy_and_convert(src_path, dest_path, format, original_format='tif'):
            """Helper function to copy and convert images to specified format."""
            if format == 'npy' and original_format == 'tif':
                data = rasterio.open(src_path).read()
                np.save(dest_path.replace('.tif', '.npy'), np.array(data))
            if format == 'tif' and original_format == 'png':
                img = Image.open(src_path)
                dest_path = dest_path.replace('.png', '.tif')
                with rasterio.open(dest_path, 'w', driver='GTiff', width=img.width, height=img.height, count=1, dtype='uint8'
                                ) as dst:
                                    array_data = np.array(img).astype('uint8')
                                    dst.write(array_data, 1)
            if format == 'npy' and original_format == 'png':
                img = Image.open(src_path)
                np.save(dest_path.replace('.png', '.npy'), np.array(img))
            else:
                shutil.copy(src_path, dest_path)

        date1, date2, season1, season2, year_type, deforestation_change, cloud1, cloud2, wasserstein, split = rest

        image1_name = f'S2A_Image_{tile_id}_{date1}_{position_1}_{position_2}.tif'
        image2_name = f'S2A_Image_{tile_id}_{date2}_{position_1}_{position_2}.tif'
        s1_image1_name = f'S1_Image_{tile_id}_{date1}_{position_1}_{position_2}.tif'
        s1_image2_name = f'S1_Image_{tile_id}_{date2}_{position_1}_{position_2}.tif'
        diff_mask_name = f'diff_{tile_id}_{date1}_{date2}_{position_1}_{position_2}_{deforestation_change}.png'
        cloud_mask_1_name = f'cloud_{tile_id}_{date1}_{position_1}_{position_2}.tif'
        cloud_mask_2_name = f'cloud_{tile_id}_{date2}_{position_1}_{position_2}.tif'

        if image1_name and image2_name:
            image1_src_path = os.path.join(images_dir, image1_name)
            image2_src_path = os.path.join(images_dir, image2_name)
            s1_image1_src_path = os.path.join(images_dir.replace('s2', 's1'), s1_image1_name)
            s1_image2_src_path = os.path.join(images_dir.replace('s2', 's1'), s1_image2_name)
            diff_mask_src_path = os.path.join(diff_mask_dir, diff_mask_name)
            cloud_mask1_src_path = os.path.join(diff_mask_dir.replace('diff_masks', 'clouds'), image1_name)
            cloud_mask2_src_path = os.path.join(diff_mask_dir.replace('diff_masks', 'clouds'), image2_name)

            image1_dest_path = os.path.join(final_dir, 'S2_A', f'{split_type}_{idx}_{image1_name}')
            image2_dest_path = os.path.join(final_dir, 'S2_B', f'{split_type}_{idx}_{image2_name}')
            s1_image1_dest_path = os.path.join(final_dir, 'S1_A', f'{split_type}_{idx}_{s1_image1_name}')
            s1_image2_dest_path = os.path.join(final_dir, 'S1_B', f'{split_type}_{idx}_{s1_image2_name}')
            diff_mask_dest_path = os.path.join(final_dir, 'masks', f'{split_type}_{idx}_{diff_mask_name}')
            cloud_mask1_dest_path = os.path.join(final_dir, 'clouds_A', f'{split_type}_{idx}_{cloud_mask_1_name}')
            cloud_mask2_dest_path = os.path.join(final_dir, 'clouds_B', f'{split_type}_{idx}_{cloud_mask_2_name}')
            
            # Copy files
            copy_and_convert(image1_src_path, image1_dest_path, save_format, original_format='tif')
            copy_and_convert(image2_src_path, image2_dest_path, save_format, original_format='tif')
            copy_and_convert(s1_image1_src_path, s1_image1_dest_path, save_format, original_format='tif')
            copy_and_convert(s1_image2_src_path, s1_image2_dest_path, save_format, original_format='tif')
            copy_and_convert(diff_mask_src_path, diff_mask_dest_path, save_format, original_format='png')
            copy_and_convert(cloud_mask1_src_path, cloud_mask1_dest_path, save_format, original_format='tif')
            copy_and_convert(cloud_mask2_src_path, cloud_mask2_dest_path, save_format, original_format='tif')
        else:
            print(f"Images for {tile_id} on dates {date1} and {date2} not found.")

    print(f'Copied {len(df_info)} images to {final_dir}')





def compute_stats_for_bands(dataset_dir, save=False, save_path=None, percentiles=[1, 99], return_values=False, file_type='npy'):
    """
    Compute percentiles, mean, and standard deviation for each band in the dataset by loading all data into memory.
    Optionally, save the computed statistics.

    Args:
    dataset_dir (str): Directory containing the dataset
    save (bool): Save the computed statistics
    save_path (str): Path to save the computed statistics
    percentiles (list): Percentiles to compute
    return_values (bool): Return the computed statistics
    file_type (str): Type of file (npy, tif)
    """
    # Gather all image files
    images_S2A = glob.glob(os.path.join(dataset_dir, 'S2_A', f'*.{file_type}'))
    images_S2B = glob.glob(os.path.join(dataset_dir, 'S2_B', f'*.{file_type}'))
    images_S1A = glob.glob(os.path.join(dataset_dir, 'S1_A', f'*.{file_type}'))
    images_S1B = glob.glob(os.path.join(dataset_dir, 'S1_B', f'*.{file_type}'))

    S2_images = images_S2A + images_S2B
    S1_images = images_S1A + images_S1B

    print(f'Found {len(S2_images)} S2 images and {len(S1_images)} S1 images')

    # Load all S2 bands from all files
    S2_bands_list = []
    for file_path in tqdm(S2_images):
        if file_type == 'npy':
            bands = np.load(file_path)
        elif file_type == 'tif':
            with rasterio.open(file_path) as src:
                bands = src.read()
        else:
            raise ValueError('Unsupported file type')

        S2_bands_list.append(bands)
        del bands


    gc.collect()

    # Stack all bands into a single array for processing
    all_S2_bands = np.stack(S2_bands_list)

    # Compute statistics across all S2 bands
    lower_S2 = np.percentile(all_S2_bands, percentiles[0], axis=(0, 2, 3))
    upper_S2 = np.percentile(all_S2_bands, percentiles[1], axis=(0, 2, 3))
    mean_S2 = np.mean(all_S2_bands, axis=(0, 2, 3))
    std_S2 = np.std(all_S2_bands, axis=(0, 2, 3))

    del all_S2_bands, S2_bands_list
    gc.collect()

    # Load all S1 bands from all files
    S1_bands_list = []
    for file_path in tqdm(S1_images):
        if file_type == 'npy':
            bands = np.load(file_path)
        elif file_type == 'tif':
            with rasterio.open(file_path) as src:
                bands = src.read()
        else:
            raise ValueError('Unsupported file type')

        S1_bands_list.append(bands)

    all_S1_bands = np.stack(S1_bands_list)
    print(all_S1_bands[:, 0, :, :].shape)
    lower_S1 = np.percentile(all_S1_bands, percentiles[0], axis=(0, 2, 3))
    upper_S1 = np.percentile(all_S1_bands, percentiles[1], axis=(0, 2, 3))
    mean_S1 = np.mean(all_S1_bands, axis=(0, 2, 3))
    std_S1 = np.std(all_S1_bands, axis=(0, 2, 3))

    if save:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        np.save(os.path.join(save_path, f'S2_{percentiles[0]}'), lower_S2)
        np.save(os.path.join(save_path, f'S2_{percentiles[1]}'), upper_S2)
        np.save(os.path.join(save_path, 'S2_mean'), mean_S2)
        np.save(os.path.join(save_path, 'S2_std'), std_S2)

        np.save(os.path.join(save_path, f'S1_{percentiles[0]}'), lower_S1)
        np.save(os.path.join(save_path, f'S1_{percentiles[1]}'), upper_S1)
        np.save(os.path.join(save_path, 'S1_mean'), mean_S1)
        np.save(os.path.join(save_path, 'S1_std'), std_S1)


    if return_values:
        return [lower_S2, upper_S2, mean_S2, std_S2, lower_S1, upper_S1, mean_S1, std_S1]
    else:
        print(f'Statistics computed and saved at {save_path}')







