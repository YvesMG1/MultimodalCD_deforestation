
import cv2
import logging
import numpy as np
import matplotlib.path as mpltPath

from shapely.geometry import Polygon
from shapely.ops import unary_union
from shapely.validation import explain_validity


def ensure_valid_polygon(poly):
    """
    Ensure that a polygon is valid. If the polygon is invalid, try to repair it.

    Args:
        poly (shapely.geometry.Polygon): The polygon to validate.
    
    Returns:  
        shapely.geometry.Polygon: The validated polygon.
    """

    if not poly.is_valid:
        # First, try simplifying the polygon
        poly = poly.simplify(0.05, preserve_topology=True)
        if not poly.is_valid:
             # If still invalid, try simplifying with a larger tolerance
            poly = poly.simplify(0.2, preserve_topology=True)
            #print(f"Polygon is invalid. Attempting to repair.")
            if not poly.is_valid:
                # As a last resort, use the convex hull
                poly = poly.convex_hull
                if not poly.is_valid:
                    print(f"Unable to repair polygon: {explain_validity(poly)}")
    return poly


def polygonize(raster_array):
    """ 
    Creates polygons from a binary raster array.
    
    Args: 
        raster_array (numpy.ndarray): The binary raster array.
    
    Returns:
        list: A list of shapely Polygon objects.
    """

    binary_image = np.where(raster_array == 1, 255, 0).astype(np.uint8)

    # Find contours from the binary image
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    polygons = []

    for contour in contours:
        if contour.shape[0] > 2:  # Check if the contour can form a polygon
            # Construct a polygon from contours
            poly = Polygon([tuple(point[0]) for point in contour])
            #poly = ensure_valid_polygon(poly) 
            path = mpltPath.Path(list(poly.exterior.coords))  # Create a path from polygon points
            
            # Create a grid of points for the whole image
            x, y = np.meshgrid(np.arange(raster_array.shape[1]), np.arange(raster_array.shape[0]))  # x and y grid arrays
            points = np.vstack((x.flatten(), y.flatten())).T

            # Create a mask where points inside the polygon are True
            grid = path.contains_points(points).reshape(raster_array.shape)
            masked_area = raster_array[grid]

            # Verify that the masked area is predominantly '1'
            if masked_area.size > 0:
                if np.mean(masked_area) > 0.5:
                    polygons.append(poly)
    
    return polygons


def calculate_aggregate_iou(pred_poly, actual_polys):
    """Calculate the intersection over union between a predicted polygon and a list of actual polygons.
    
    Args:
        pred_poly (shapely.geometry.Polygon): The predicted polygon.
        actual_polys (list): A list of actual polygons.
    
    Returns:
        float: The intersection over union score.
        list: A list of actual polygons that intersect with the predicted polygon.
    """


    matching_polys = [poly for poly in actual_polys if pred_poly.intersects(poly)]

    if not matching_polys:
        return 0, []

    combined_actual_poly = unary_union(matching_polys)
    combined_actual_poly = ensure_valid_polygon(combined_actual_poly)

    if not combined_actual_poly.is_valid:
        print("Combined polygon is still invalid.")
        return 0, []

    intersection = pred_poly.intersection(combined_actual_poly).area
    union = pred_poly.union(combined_actual_poly).area
    return intersection / union, matching_polys


class PolygonConfuseMatrixMeter():
    """Manages the polygon-based confusion matrix for deforestation analysis."""
    def __init__(self, iou_threshold=0.5):
        self.iou_threshold = iou_threshold
        self.reset()

    def update(self, predicted_polys, actual_polys):
        cm, ious = self.polygon_cm(predicted_polys, actual_polys)
        self.ious.extend(ious)
        for key in self.conf_matrix:
            self.conf_matrix[key] += cm[key]

    def get_scores(self):
        """Calculate class-specific metrics and return them in a dictionary."""
        hist = self.get_cm()
        n_class = hist.shape[0]
        tp = np.diag(hist)
        sum_a1 = hist.sum(axis=1)
        sum_a0 = hist.sum(axis=0)
        
        # Calculate metrics
        recall = tp / (sum_a1 + np.finfo(np.float32).eps)
        precision = tp / (sum_a0 + np.finfo(np.float32).eps)
        f1 = 2 * recall * precision / (recall + precision + np.finfo(np.float32).eps)
        mean_f1 = np.nanmean(f1)

        # Prepare class-specific scores
        cls_precision = dict(zip(['precision_'+str(i) for i in range(n_class)], precision))
        cls_recall = dict(zip(['recall_'+str(i) for i in range(n_class)], recall))
        cls_f1 = dict(zip(['F1_'+str(i) for i in range(n_class)], f1))

        score_dict = {
            'mf1': mean_f1,
            'iou': np.mean(self.ious) if self.ious else 0,
            'acc': hist.trace() / hist.sum(),
        }  
        score_dict.update(cls_f1)
        score_dict.update(cls_precision)
        score_dict.update(cls_recall)

        return score_dict

    def polygon_cm(self, predicted_polys, actual_polys):
        """Calculate a confusion matrix for polygon predictions."""
        cm = {'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0}
        ious = []

        #TN 1 if no predicted polygons and no actual polygons
        if len(predicted_polys) == 0 and len(actual_polys) == 0:
            cm['TN'] += 1
            return cm, ious
        

        matched = []
        matched_iou0 = []

        # ensure valid polygons
        valid_predicted_polys = [ensure_valid_polygon(poly) for poly in predicted_polys]
        valid_actual_polys = [ensure_valid_polygon(poly) for poly in actual_polys]

        for pred_poly in valid_predicted_polys:
            iou, matched_polys = calculate_aggregate_iou(pred_poly, valid_actual_polys)
            matched_iou0.extend(matched_polys)
            if iou >= self.iou_threshold:
                cm['TP'] += len(matched_polys)
                matched.extend(matched_polys)
            else:
                cm['FP'] += 1
            ious.append(iou)
        
        for act_poly in valid_actual_polys:
            if act_poly not in matched:
                cm['FN'] += 1
            if act_poly not in matched_iou0:
                ious.append(0)

        return cm, ious
    
    def reset(self):
        """Resets all attributes to their initial configuration, including IoU scores."""
        self.conf_matrix = {'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0}
        self.ious = []

    def get_cm(self):
        cm = np.array([[self.conf_matrix['TP'], self.conf_matrix['FP']],
                          [self.conf_matrix['FN'], self.conf_matrix['TN']]])
        return cm




