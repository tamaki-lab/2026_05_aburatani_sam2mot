"""
MOT result writer.
"""
from typing import List
import os

def write_trajectories(file_path: str, trajectories: List[dict]):
    """
    Writes trajectories to a MOT TrackEval compatible format.
    Format: frame, track_id, x, y, w, h, score, -1, -1, -1
    
    Args:
        file_path: Output file path.
        trajectories: List of dicts, each containing:
            {'frame_id': int, 'track_id': int, 'bbox_xywh': List[float], 'score': float}
    """
    # Create parent directories if they don't exist
    parent_dir = os.path.dirname(file_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    
    with open(file_path, 'w') as f:
        for traj in trajectories:
            frame_id = traj['frame_id']
            track_id = traj['track_id']
            x, y, w, h = traj['bbox_xywh']
            score = traj.get('score', 1.0)
            
            line = f"{frame_id},{track_id},{x:.2f},{y:.2f},{w:.2f},{h:.2f},{score:.4f},-1,-1,-1\n"
            f.write(line)
