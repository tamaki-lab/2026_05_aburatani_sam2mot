"""
Track dataclass for SAM2MOT-lite.
"""
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np

# Track states
STATE_RELIABLE = "reliable"
STATE_PENDING = "pending"
STATE_SUSPICIOUS = "suspicious"
STATE_LOST = "lost"


@dataclass
class Track:
    """
    Represents a single tracked object.

    Attributes:
        track_id: Unique integer ID (1, 2, 3, ...).
        state: One of reliable / pending / suspicious / lost.
        mask: Binary mask (H, W) for the current frame, or None.
        bbox: Bounding box [x1, y1, x2, y2] (xyxy) derived from mask, or None.
        score: SAM2 provisional confidence score for the current frame.
        score_history: List of past scores (oldest first).
        lost_count: Number of consecutive frames with no valid mask.
        last_frame: MOT frame_id (1-indexed) of the last successful update.
        keyframe_idx: SAM2 frame_idx (0-indexed) of the last box prompt.
        active: Whether this track is still alive.
        corrupted: Whether this track is marked as corrupted due to conflicts.
    """
    track_id: int
    state: str = STATE_RELIABLE
    mask: Optional[np.ndarray] = None
    bbox: Optional[List[float]] = None
    score: float = 0.0
    score_history: List[float] = field(default_factory=list)
    lost_count: int = 0
    last_frame: int = 0
    keyframe_idx: int = 0
    active: bool = True
    corrupted: bool = False
