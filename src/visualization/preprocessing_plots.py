import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import rasterio

sys.path.append('/scratch/yves/Thesis/Yves-MSc-Thesis/src/data/preprocessing')
from utils import parse_filename, get_files


def percentile_stretch(band, lower_percentile=1, upper_percentile=98):
    """Apply percentile stretch to an image band."""

    min_val, max_val = np.percentile(band, [lower_percentile, upper_percentile])
    band = np.clip(band, min_val, max_val)
    band = (band - min_val) / (max_val - min_val) * 255

    return band.astype(np.uint8)

def plot_diff_mask(diff_mask_path, band = "rgb", s1 = True, s1_band = 1, normalization = 'percentile_stretch', 
                   percentile = [1, 98], diff_top_percentile = 50):
    

    tile_id, date_1, date_2, position_1, position_2, _ = parse_filename(os.path.basename(diff_mask_path), mask_type='diff_mask')

    # get basepath of the s2 images
    basepath = os.path.dirname(diff_mask_path).replace('diff_masks', 's2')
    image_name_1 = f"S2A_Image_{tile_id}_{date_1}_{position_1}_{position_2}.tif"
    image_name_2 = f"S2A_Image_{tile_id}_{date_2}_{position_1}_{position_2}.tif"
    image_path_1 = os.path.join(basepath, image_name_1)
    image_path_2 = os.path.join(basepath, image_name_2)

    if s1:
        basepath = os.path.dirname(diff_mask_path).replace('diff_masks', 's1')
        s1_name_1 = f"S1_Image_{tile_id}_{date_1}_{position_1}_{position_2}.tif"
        s1_name_2 = f"S1_Image_{tile_id}_{date_2}_{position_1}_{position_2}.tif"
        s1_path_1 = os.path.join(basepath, s1_name_1)
        s1_path_2 = os.path.join(basepath, s1_name_2)


    with rasterio.open(image_path_1) as src1, rasterio.open(image_path_2) as src2:
        
        # Read the RGB bands
        red1 = src1.read(1)
        green1 = src1.read(2)
        blue1 = src1.read(3)

        red2 = src2.read(1)
        green2 = src2.read(2)
        blue2 = src2.read(3)

        if band == "rgb":
            if normalization == 'percentile_stretch':
                red_stretched1 = percentile_stretch(red1, lower_percentile=percentile[0], upper_percentile=percentile[1])
                green_stretched1 = percentile_stretch(green1, lower_percentile=percentile[0], upper_percentile=percentile[1])
                blue_stretched1 = percentile_stretch(blue1, lower_percentile=percentile[0], upper_percentile=percentile[1])

                red_stretched2 = percentile_stretch(red2, lower_percentile=percentile[0], upper_percentile=percentile[1])
                green_stretched2 = percentile_stretch(green2, lower_percentile=percentile[0], upper_percentile=percentile[1])
                blue_stretched2 = percentile_stretch(blue2, lower_percentile=percentile[0], upper_percentile=percentile[1])
            
            else:
                red_stretched1 = red1
                green_stretched1 = green1
                blue_stretched1 = blue1

                red_stretched2 = red2
                green_stretched2 = green2
                blue_stretched2 = blue2

            rgb1 = np.dstack((red_stretched1, green_stretched1, blue_stretched1))
            rgb2 = np.dstack((red_stretched2, green_stretched2, blue_stretched2))

        else:
            if normalization == 'percentile_stretch':
                s2_band_1 = percentile_stretch(src1.read(band), lower_percentile=percentile[0], upper_percentile=percentile[1])
                s2_band_2 = percentile_stretch(src2.read(band), lower_percentile=percentile[0], upper_percentile=percentile[1])
            else:
                s2_band_1 = src1.read(band)
                s2_band_2 = src2.read(band)

        if s1:
            with rasterio.open(s1_path_1) as s1_src1, rasterio.open(s1_path_2) as s1_src2:
                if normalization == 'percentile_stretch':
                    s1_band_1 = percentile_stretch(s1_src1.read(s1_band), lower_percentile=percentile[0], upper_percentile=percentile[1])
                    s1_band_2 = percentile_stretch(s1_src2.read(s1_band), lower_percentile=percentile[0], upper_percentile=percentile[1])
                else:
                    s1_band_1 = s1_src1.read(s1_band)
                    s1_band_2 = s1_src2.read(s1_band)

                lower_percentile = 100 - diff_top_percentile
                upper_percentile = diff_top_percentile

                diff_band = s1_band_2 - s1_band_1
                lower_bound = np.percentile(diff_band, lower_percentile)

                
                # Apply masking
                # Mask out values outside the lower and upper bounds
                mask = (diff_band <= lower_bound) 
                s1_diff_band = np.where(mask, diff_band, 0)

        # Read the diff mask
        diff_mask = np.array(Image.open(diff_mask_path))

        if band == "rgb":
            overlay = rgb1.copy()
            overlay[:, :, 0] = np.where(diff_mask, 255, red_stretched1)
            overlay[:, :, 1] = np.where(diff_mask, 0, green_stretched1)
            overlay[:, :, 2] = np.where(diff_mask, 0, blue_stretched1)
        else:
            overlay = diff_mask
        
        # Determine number of subplots
        if band == "rgb" and s1 is False:
            fig, ax = plt.subplots(1, 3, figsize=(18, 6))
            ax[0].imshow(rgb1)
            ax[0].axis('off')
            ax[0].set_title('Image 1')

            ax[1].imshow(rgb2)
            ax[1].axis('off')
            ax[1].set_title('Image 2')

            ax[2].imshow(overlay)
            ax[2].axis('off')
            ax[2].set_title('Diff Mask Overlay')
            plt.show()

        elif band == "rgb" and s1 is True:
            fig, ax = plt.subplots(2, 3, figsize=(18, 6))
            ax[0, 0].imshow(rgb1)
            ax[0, 0].axis('off')
            ax[0, 0].set_title('Image 1')

            ax[0, 1].imshow(rgb2)
            ax[0, 1].axis('off')
            ax[0, 1].set_title('Image 2')

            ax[0, 2].imshow(overlay)
            ax[0, 2].axis('off')
            ax[0, 2].set_title('Diff Mask Overlay')

            ax[1, 0].imshow(s1_band_1, cmap='gray')
            ax[1, 0].axis('off')
            ax[1, 0].set_title(f'S1 Band {s1_band} for Image 1')

            ax[1, 1].imshow(s1_band_2, cmap='gray')
            ax[1, 1].axis('off')
            ax[1, 1].set_title(f'S1 Band {s1_band} for Image 2')
            
            ax[1, 2].imshow(s1_diff_band, cmap='gray')
            ax[1, 2].axis('off')
            ax[1, 2].set_title(f'S1 Diff Band {s1_band}')

            plt.show()
        
        elif band != "rgb" and s1 is False:
            fig, ax = plt.subplots(1, 3, figsize=(18, 6))
            ax[0].imshow(s2_band_1, cmap='RdYlGn')
            ax[0].axis('off')
            ax[0].set_title(f'S2 Band {band} for Image 1')

            ax[1].imshow(s2_band_2, cmap='RdYlGn')
            ax[1].axis('off')
            ax[1].set_title(f'S2 Band {band} for Image 2')

            ax[2].imshow(overlay)
            ax[2].axis('off')
            ax[2].set_title('Diff Mask Overlay')

            plt.show()

        elif band != "rgb" and s1 is True:

            fig, ax = plt.subplots(2, 3, figsize=(18, 6))
            ax[0, 0].imshow(s2_band_1, cmap='RdYlGn')
            ax[0, 0].axis('off')
            ax[0, 0].set_title(f'S2 Band {band} for Image 1')
            
            ax[0, 1].imshow(s2_band_2, cmap='RdYlGn')
            ax[0, 1].axis('off')
            ax[0, 1].set_title(f'S2 Band {band} for Image 2')

            ax[0, 2].imshow(overlay)
            ax[0, 2].axis('off')
            ax[0, 2].set_title('Diff Mask Overlay')

            ax[1, 0].imshow(s1_band_1, cmap='gray')
            ax[1, 0].axis('off')
            ax[1, 0].set_title(f'S1 Band {s1_band} for Image 1')

            ax[1, 1].imshow(s1_band_2, cmap='gray')
            ax[1, 1].axis('off')
            ax[1, 1].set_title(f'S1 Band {s1_band} for Image 2')

            ax[1, 2].imshow(s1_diff_band, cmap='gray')
            ax[1, 2].axis('off')
            ax[1, 2].set_title(f'S1 Diff Band {s1_band}')

            plt.show()
        
        else:
            print('Error: Invalid combination of RGB and additional bands')



def plot_images(image_path, use_mask = False, mask_path = None, use_rgb = True, additional_bands = None, 
                use_s1 = False, s1_band = 1, normalization = 'percentile_stretch', percentile = [1, 98],
                titles = None):
    
    """Plot the RGB image and additional bands with optional mask overlay

    Args:
        image_path (str): The path to the image file.
        use_mask (bool): Whether to use a mask overlay. Default is False.
        mask_path (str): The path to the mask file. Default is None.
        use_rgb (bool): Whether to use the RGB bands. Default is True.
        additional_bands (list): The list of additional bands to plot. Default is None.
        use_s1 (bool): Whether to use Sentinel-1 bands. Default is False.
        s1_band (int): The Sentinel-1 band to plot. Default is 1.
        normalization (str): The normalization method to use. Default is 'percentile_stretch'.
        percentile (list): The lower and upper percentiles for percentile stretch. Default is [1, 98].
        titles (list): The list of titles for the additional bands. Default is None.
    """

    try:
        with rasterio.open(image_path) as src:

            if use_rgb:
                red = src.read(1)
                green = src.read(2)
                blue = src.read(3)

                if normalization == 'percentile_stretch':
                    red_stretched = percentile_stretch(red, lower_percentile=percentile[0], upper_percentile=percentile[1])
                    green_stretched = percentile_stretch(green, lower_percentile=percentile[0], upper_percentile=percentile[1])
                    blue_stretched = percentile_stretch(blue, lower_percentile=percentile[0], upper_percentile=percentile[1])

                else:
                    red_stretched = red
                    green_stretched = green
                    blue_stretched = blue

                rgb_img = np.dstack((red_stretched, green_stretched, blue_stretched))
  
            if additional_bands is not None:
                bands = [src.read(band) for band in additional_bands]
                if normalization == 'percentile_stretch':
                    # loop through the bands and apply percentile stretch
                    bands_stretched = [percentile_stretch(band, lower_percentile=percentile[0], upper_percentile=percentile[1]) for band in bands]
                else:
                    bands_stretched = bands
            
            # Mask
            if use_mask:
                # find the mask path
                if mask_path is None:
                    print('No mask path provided')
                
                mask = np.array(Image.open(mask_path))

                if use_rgb:
                    overlay = rgb_img.copy()
                    overlay[:, :, 0] = np.where(mask, 255, red_stretched)
                    overlay[:, :, 1] = np.where(mask, 0, green_stretched)
                    overlay[:, :, 2] = np.where(mask, 0, blue_stretched)
                else:
                    overlay = mask

            if use_s1:
                s1_path = image_path.replace('s2', 's1').replace('S2A', 'S1')
                with rasterio.open(s1_path) as s1_src:
                    s1_band = s1_src.read(s1_band)
                    if normalization == 'percentile_stretch':
                        s1_band_stretched = percentile_stretch(s1_band, lower_percentile=percentile[0], upper_percentile=percentile[1])
                    else:
                        s1_band_stretched = s1_band


            # Determine number of subplots
            if use_rgb is True and use_mask is False and additional_bands is None and use_s1 is False:
                fig, ax = plt.subplots(1, 1, figsize=(4, 4))
                ax.imshow(rgb_img)
                ax.axis('off')
                ax.set_title('Image')
                plt.show()

            elif use_rgb is True and use_mask is True and additional_bands is None and use_s1 is False:
                fig, ax = plt.subplots(1, 2, figsize=(10, 6))
                ax[0].imshow(rgb_img)
                ax[0].axis('off')
                ax[0].set_title('Image')
                
                ax[1].imshow(overlay)
                ax[1].axis('off')
                ax[1].set_title('Mask Overlay')
                plt.show()

            elif use_rgb is True and use_mask is False and additional_bands is not None and use_s1 is False:
                fig, ax = plt.subplots(1, len(additional_bands) + 1, figsize=(12, 6))
                ax[0].imshow(rgb_img)
                ax[0].axis('off')
                ax[0].set_title('Image')

                for i in range(len(additional_bands)):
                    ax[i + 1].imshow(bands_stretched[i])
                    ax[i + 1].axis('off')
                    ax[i + 1].set_title(f'Band {additional_bands[i]}')
                plt.show()
            
            elif use_rgb is True and use_mask is True and additional_bands is not None and use_s1 is False:
                fig, ax = plt.subplots(1, len(additional_bands) + 2, figsize=(18, 6))
                ax[0].imshow(rgb_img)
                ax[0].axis('off')
                ax[0].set_title('Image')

                ax[1].imshow(overlay)
                ax[1].axis('off')
                ax[1].set_title('Mask Overlay')

                for i in range(len(additional_bands)):
                    ax[i + 2].imshow(bands_stretched[i])
                    ax[i + 2].axis('off')
                    ax[i + 2].set_title(f'Band {additional_bands[i]}')
                plt.show()

            elif use_rgb is False and use_mask is False and additional_bands is not None and use_s1 is False:
                fig, ax = plt.subplots(1, len(additional_bands), figsize=(12, 6))
                for i in range(len(additional_bands)):
                    ax[i].imshow(bands_stretched[i])
                    ax[i].axis('off')
                    ax[i].set_title(f'Band {additional_bands[i]}')
                plt.show()
            
            elif use_rgb is False and use_mask is True and additional_bands is not None and use_s1 is False:
                fig, ax = plt.subplots(1, len(additional_bands) + 1, figsize=(18, 6))

                for i in range(len(additional_bands)):
                    ax[i].imshow(bands_stretched[i])
                    ax[i].axis('off')
                    ax[i].set_title(f'Band {additional_bands[i]}')

                ax[-1].imshow(overlay)
                ax[-1].axis('off')
                ax[-1].set_title('Mask Overlay')
                plt.show()

            elif use_rgb is True and use_mask is False and additional_bands is None and use_s1 is True:
                fig, ax = plt.subplots(1, 2, figsize=(12, 6))
                ax[0].imshow(rgb_img)
                ax[0].axis('off')
                ax[0].set_title('Image')

                ax[1].imshow(s1_band_stretched, cmap='gray')
                ax[1].axis('off')
                ax[1].set_title(f'S1 Band {s1_band}')
                plt.show()
            
            elif use_rgb is True and use_mask is True and additional_bands is None and use_s1 is True:

                fig, ax = plt.subplots(1, 3, figsize=(18, 6))
                ax[0].imshow(rgb_img)
                ax[0].axis('off')
                ax[0].set_title('Image')

                ax[1].imshow(s1_band_stretched, cmap='gray')
                ax[1].axis('off')
                ax[1].set_title(f'S1 Band {s1_band}')

                ax[2].imshow(overlay)
                ax[2].axis('off')
                ax[2].set_title('Mask Overlay')
                plt.show()
            
            elif use_rgb is True and use_mask is False and additional_bands is not None and use_s1 is True:
                
                fig, ax = plt.subplots(1, len(additional_bands) + 2, figsize=(18, 6))
                ax[0].imshow(rgb_img)
                ax[0].axis('off')
                ax[0].set_title(titles[0]) if titles is not None else ax[0].set_title('Image')

                for i in range(len(additional_bands)):
                    ax[i + 1].imshow(bands_stretched[i])
                    ax[i + 1].axis('off')
                    ax[i + 1].set_title(f'Band {additional_bands[i]}') if titles is None else ax[i + 1].set_title(titles[i])

                ax[len(additional_bands) + 1].imshow(s1_band_stretched, cmap='gray')
                ax[len(additional_bands) + 1].axis('off')
                ax[len(additional_bands) + 1].set_title('S1') if titles is None else ax[len(additional_bands) + 1].set_title(titles[-1])
                
                plt.show()
                

            
            else:
                print('Error: Invalid combination of RGB, mask and additional bands')
    except: 
        print('Error: Could not open image file')


def plot_cloud_mask(image_path, normalization = 'percentile_stretch', percentile = [1, 98],
                    overlay_intensity = 0.5, white_intensity = 255, cloud_upper = 0.1, cloud_lower = 0.01):

    """Plot the RGB image and the cloud mask overlay
    
    Args:
        image_path (str): The path to the image file.
        normalization (str): The normalization method to use. Default is 'percentile_stretch'.
        percentile (list): The lower and upper percentiles for percentile stretch. Default is [1, 98].
        overlay_intensity (float): The intensity of the overlay. Default is 0.5.
        white_intensity (int): The intensity of the white pixels in the overlay. Default is 255.
        cloud_upper (float): The upper threshold for cloud coverage. Default is 0.1.
        cloud_lower (float): The lower threshold for cloud coverage. Default is 0.01.
    
    Returns:
        tuple: The figure and axes objects.
    """



    cloud_path = image_path.replace('s2', 'clouds')
    with rasterio.open(image_path) as src, rasterio.open(cloud_path) as cloud_src:

        mask = cloud_src.read(1)
        cloud_percentage = np.count_nonzero(mask) / mask.size
        if cloud_percentage <= cloud_upper and cloud_percentage >= cloud_lower:

            # Read the RGB bands
            red = src.read(1)
            green = src.read(2)
            blue = src.read(3)

            if normalization == 'percentile_stretch':
                red_stretched = percentile_stretch(red, lower_percentile=percentile[0], upper_percentile=percentile[1])
                green_stretched = percentile_stretch(green, lower_percentile=percentile[0], upper_percentile=percentile[1])
                blue_stretched = percentile_stretch(blue, lower_percentile=percentile[0], upper_percentile=percentile[1])

            rgb = np.dstack((red_stretched, green_stretched, blue_stretched))

            # Read the cloud mask and create overlay
            mask = cloud_src.read(1)

            overlay = rgb.copy()
        
            overlay[:, :, 0] = np.where(mask, white_intensity, red_stretched)
            overlay[:, :, 1] = np.where(mask, green_stretched * (1 - mask * overlay_intensity), green_stretched)
            overlay[:, :, 2] = np.where(mask, blue_stretched * (1 - mask * overlay_intensity), blue_stretched)

            # Plot the image and the cloud mask overlay
            fig, ax = plt.subplots(1, 3, figsize=(12, 6))
            ax[0].imshow(rgb)
            ax[0].axis('off')
            ax[0].set_title('Image')

            ax[1].imshow(mask, cmap='gray')
            ax[1].axis('off')
            ax[1].set_title('Cloud Mask')

            ax[2].imshow(overlay)
            ax[2].axis('off')
            ax[2].set_title('Cloud Mask Overlay')

            return fig, ax

        else:
            return None


                    




        




