
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection
import torch


def percentile_stretch(image, lower_percentile, upper_percentile):
    """Normalize image using percentile stretching.
    Args:
        image_band: np.array
        lower_percentile: int
        upper_percentile: int
    
    Returns: 
        np.array: Normalized image_band
    """

    lower, upper = np.percentile(image, [lower_percentile, upper_percentile])
    normalized_image = np.clip((image - lower) / (upper - lower), 0, 1)
    return normalized_image


def process_image(image, band, percentiles, normalize=True):
    """
    Process the image for visualization.

    Args:
    image (torch.Tensor): 3D tensor of the image.
    band (str or int): The band to visualize. Options: 'rgb', [4, 5, 6, 7]
    percentiles (list): The lower and upper percentiles to use for percentile normalization.
    normalize (bool): Whether to normalize the image.

    Returns:
    np.array: The processed band of the image. 
    """

    if band == 'rgb':
        if isinstance(image, torch.Tensor):
            channels = image[:3].cpu().detach().numpy().transpose(1, 2, 0)
        else:
            channels = image[:3].transpose(1, 2, 0)
        if normalize:
            return percentile_stretch(channels, percentiles[0], percentiles[1])
    else:
        if isinstance(image, torch.Tensor):
            image = image.cpu().detach().numpy()
        channel = image[band]
        if normalize:
            return percentile_stretch(channel, percentiles[0], percentiles[1])
        return channel



def plot_polygons(predicted_polys, actual_polys, title="Polygon Comparison Plot", image_size = (128, 128)):
    """
    Plot the predicted and actual polygons on the image.

    Args:
    predicted_polys (list): A list of predicted polygons.
    actual_polys (list): A list of actual polygons.
    title (str): The title of the plot.
    image_size (tuple): The size of the image (rows, columns).

    Returns:
    plot: A plot of the predicted and actual polygons.
    """

    fig, ax = plt.subplots()
    predicted_patches = []
    actual_patches = []
    
    # Create a patch for each predicted polygon and add to the list of patches
    for poly in predicted_polys:
        if not poly.is_empty:
            predicted_patch = MplPolygon(list(poly.exterior.coords), closed=True, edgecolor='red', fill=False, linewidth=2)
            predicted_patches.append(predicted_patch)
    
    # Create a patch for each actual polygon and add to the list of patches
    for poly in actual_polys:
        if not poly.is_empty:
            actual_patch = MplPolygon(list(poly.exterior.coords), closed=True, edgecolor='blue', fill=False, linewidth=2)
            actual_patches.append(actual_patch)
    
    # Create collections for the patches
    predicted_collection = PatchCollection(predicted_patches, match_original=True)
    actual_collection = PatchCollection(actual_patches, match_original=True)
    
    # Add collections to the axes
    ax.add_collection(predicted_collection)
    ax.add_collection(actual_collection)
    
    # Set plot limits and properties
    ax.set_xlim([0, image_size[0]])  # Set these limits based on your actual data
    ax.set_ylim([0, image_size[1]])  # Set these limits based on your actual data
    ax.set_aspect('equal')
    plt.title(title)
    plt.legend([predicted_collection, actual_collection], ['Predicted', 'Actual'])
    plt.show()


def plot_conf_matrix(cm, save_path=None, save=False, show=True, model_name = None):

    """
    Plot the confusion matrix for a model.

    Args:
        cm (np.array): The confusion matrix.
        save_path (str): The path to save the plot.
        save (bool): Whether to save the plot.
        show (bool): Whether to display the plot.
        model_name (str): The name of the model.
        fold (int): The fold number.
        data_type (str): The type of data used to generate the confusion matrix. Options: 'pixel', 'polygon'
    
    Returns:
        plot: A plot of the confusion matrix.

    """

    # Plot confusion matrix
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    axes[0].matshow(cm, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            axes[0].text(x=j, y=i, s=cm[i, j], va='center', ha='center', size='xx-large')
    axes[0].set_title(f"Confusion Matrix for {model_name}")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")

    # Normalized confusion matrix
    axes[1].matshow(cm / (cm.sum(axis=1) + np.finfo(float).eps), cmap=plt.cm.Blues, alpha=0.3)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            axes[1].text(x=j, y=i, s=round(cm[i, j] / (cm.sum(axis=1)[i] + np.finfo(float).eps), 2), va='center', ha='center', size='xx-large')
    axes[1].set_title(f"Normalised Confusion Matrix for {model_name}")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Actual")

    plt.tight_layout()
    
    if save:
        plt.savefig(save_path)

    if show:
        plt.show()


def plot_loss(train_loss, val_loss, show=True, save=False, save_path=None, model_name=None):

    """
    Plot the training and validation loss for a model.
    
    Args:
        train_loss (list): A list of training loss values.
        val_loss (list): A list of validation loss values.
        show (bool): Whether to display the plot.
        save (bool): Whether to save the plot.
        save_path (str): The path to save the plot.
        model_name (str): The name of the model.
        fold (int): The fold number.
    
    Returns:
        plot: A plot of the training and validation loss.
    """

    epochs = range(1, len(train_loss) + 1)

    fig, ax = plt.subplots()
    ax.plot(epochs, np.log10(train_loss), 'b-', label='Train Loss')
    ax.plot(epochs, np.log10(val_loss), 'r-', label='Validation Loss')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.set_title(f"Loss Plot for {model_name}")

    if save:
        plt.savefig(save_path)

    if show:
        plt.show()