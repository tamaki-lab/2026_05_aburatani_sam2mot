"""
Matching utilities.
"""
from typing import List, Tuple
import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    linear_sum_assignment = None

def iou_matching(
    tracks: List[List[float]], 
    detections: List[List[float]], 
    iou_thr: float = 0.5
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """
    Matches tracks (xyxy) to detections (xyxy) using Hungarian matching on IoU cost.
    
    Returns:
        matches: List of tuples (track_idx, det_idx)
        unmatched_tracks: List of indices of unmatched tracks
        unmatched_detections: List of indices of unmatched detections
    """
    if linear_sum_assignment is None:
        raise ImportError("scipy is required for matching. Please install scipy: pip install scipy")
        
    if not tracks or not detections:
        return [], list(range(len(tracks))), list(range(len(detections)))
        
    from tracker.mask_utils import bbox_iou
    
    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float32)
    
    for t, trk in enumerate(tracks):
        for d, det in enumerate(detections):
            cost_matrix[t, d] = 1.0 - bbox_iou(trk, det)
            
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    matches = []
    unmatched_tracks = set(range(len(tracks)))
    unmatched_detections = set(range(len(detections)))
    
    for r, c in zip(row_ind, col_ind):
        iou = 1.0 - cost_matrix[r, c]
        if iou >= iou_thr:
            matches.append((r, c))
            unmatched_tracks.discard(r)
            unmatched_detections.discard(c)
            
    return matches, list(unmatched_tracks), list(unmatched_detections)
