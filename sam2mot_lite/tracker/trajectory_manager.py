"""
Trajectory manager for SAM2MOT-lite.

Milestone 4: Minimal version.
- Initializes tracks from first-frame detections via SAM2 box prompts.
- Updates tracks with propagation results.
- Provides active track list.

M5+ will add: object addition, removal, quality reconstruction, COI.
"""
from typing import Dict, List, Optional, Tuple
import numpy as np

from tracker.track import Track, STATE_RELIABLE
from tracker.detection import Detection


class TrajectoryManager:
    """Manages the lifecycle of Track objects."""

    def __init__(self, config: dict):
        self.config = config
        self.tracks: List[Track] = []
        self._next_id: int = 1

    @property
    def next_track_id(self) -> int:
        """Next available track ID (1-indexed)."""
        return self._next_id

    # ------------------------------------------------------------------
    # Track creation
    # ------------------------------------------------------------------

    def create_track(
        self,
        frame_id: int,
        mask: Optional[np.ndarray],
        bbox: Optional[List[float]],
        score: float,
        keyframe_idx: int,
    ) -> Track:
        """Create a new Track and register it."""
        track = Track(
            track_id=self._next_id,
            state=STATE_RELIABLE,
            mask=mask,
            bbox=bbox,
            score=score,
            score_history=[score],
            lost_count=0,
            last_frame=frame_id,
            keyframe_idx=keyframe_idx,
            active=True,
        )
        self._next_id += 1
        self.tracks.append(track)
        return track

    # ------------------------------------------------------------------
    # Bulk initialisation from detections
    # ------------------------------------------------------------------

    def init_tracks_from_detections(
        self,
        detections: List[Detection],
        wrapper,  # SAM2Wrapper (lazy import to avoid SAM2 dep in unit tests)
    ) -> List[Track]:
        """
        Create tracks from a list of detections by sending box prompts
        to SAM2.  Returns the list of successfully created Track objects.
        """
        from tracker.sam2_wrapper import mot_to_sam2_frame
        created: List[Track] = []
        for det in detections:
            frame_idx = mot_to_sam2_frame(det.frame_id)
            obj_id = self._next_id  # peek – create_track will increment
            mask, bbox, score = wrapper.add_box_prompt(
                frame_idx, obj_id, det.box_xyxy
            )
            if mask is not None:
                track = self.create_track(
                    frame_id=det.frame_id,
                    mask=mask,
                    bbox=bbox,
                    score=score,
                    keyframe_idx=frame_idx,
                )
                created.append(track)
        return created

    # ------------------------------------------------------------------
    # Per-frame update
    # ------------------------------------------------------------------

    def update_track(
        self,
        track: Track,
        frame_id: int,
        mask: Optional[np.ndarray],
        bbox: Optional[List[float]],
        score: float,
    ) -> None:
        """Update a track with new frame results from SAM2 propagation."""
        track.mask = mask
        track.bbox = bbox
        track.score = score
        track.score_history.append(score)
        track.last_frame = frame_id

        if mask is not None:
            track.lost_count = 0
        else:
            track.lost_count += 1

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_active_tracks(self) -> List[Track]:
        """Return all currently active tracks."""
        return [t for t in self.tracks if t.active]

    def get_track_by_id(self, track_id: int) -> Optional[Track]:
        """Lookup a track by its ID.  Returns None if not found."""
        for t in self.tracks:
            if t.track_id == track_id:
                return t
        return None
