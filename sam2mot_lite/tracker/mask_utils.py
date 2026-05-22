"""
Mask and bounding box utilities.
"""
import numpy as np
from typing import List, Optional

def mask_to_box(mask: np.ndarray) -> Optional[List[float]]:
    """
    Computes bounding box (xyxy) from a binary mask.
    Returns None if mask is empty.
    """
    if mask.sum() == 0:
        return None
    
    y_indices, x_indices = np.where(mask > 0)
    x_min, x_max = x_indices.min(), x_indices.max()
    y_min, y_max = y_indices.min(), y_indices.max()
    
    return [float(x_min), float(y_min), float(x_max), float(y_max)]

def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """
    Computes Intersection over Union (IoU) between two binary masks.
    """
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    
    if union == 0:
        return 0.0
    return float(intersection / union)

def union_masks(masks: List[np.ndarray]) -> Optional[np.ndarray]:
    """
    Computes the union of a list of binary masks.
    Returns None if list is empty.
    """
    if not masks:
        return None
        
    union = masks[0].copy()
    for m in masks[1:]:
        union = np.logical_or(union, m)
    return union

def bbox_iou(box_a: List[float], box_b: List[float]) -> float:
    """
    Computes Intersection over Union (IoU) between two bounding boxes (xyxy).
    """
    x1_a, y1_a, x2_a, y2_a = box_a
    x1_b, y1_b, x2_b, y2_b = box_b
    
    x1_inter = max(x1_a, x1_b)
    y1_inter = max(y1_a, y1_b)
    x2_inter = min(x2_a, x2_b)
    y2_inter = min(y2_a, y2_b)
    
    if x2_inter < x1_inter or y2_inter < y1_inter:
        return 0.0
        
    inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    area_a = (x2_a - x1_a) * (y2_a - y1_a)
    area_b = (x2_b - x1_b) * (y2_b - y1_b)
    
    union_area = area_a + area_b - inter_area
    if union_area == 0:
        return 0.0
        
    return float(inter_area / union_area)

def compute_free_area_ratio(box: List[float], free_mask: np.ndarray) -> float:
    """
    Computes the ratio of free area inside the bounding box.
    """
    x1, y1, x2, y2 = map(int, map(round, box))
    
    # Clip coordinates to image boundaries
    h, w = free_mask.shape
    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w - 1))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h - 1))
    
    if x1 >= x2 or y1 >= y2:
        return 0.0
        
    box_area = (x2 - x1) * (y2 - y1)
    if box_area == 0:
        return 0.0
        
    free_pixels = free_mask[y1:y2, x1:x2].sum()
    return float(free_pixels / box_area)
