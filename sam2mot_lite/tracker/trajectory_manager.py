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

from tracker.track import (
    Track,
    STATE_RELIABLE,
    STATE_PENDING,
    STATE_SUSPICIOUS,
    STATE_LOST,
)
from tracker.detection import Detection
from tracker.matching import iou_matching
from tracker.mask_utils import union_masks, compute_free_area_ratio, mask_iou


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

        # 1. State Transition
        reliable_thr = self.config.get("reliable_thr", 2.0)
        pending_thr = self.config.get("pending_thr", 0.0)
        lost_thr = self.config.get("lost_thr", -2.0)
        lost_tolerance = self.config.get("lost_tolerance", 25)

        if mask is None:
            track.state = STATE_LOST
        else:
            if score >= reliable_thr:
                track.state = STATE_RELIABLE
            elif score >= pending_thr:
                track.state = STATE_PENDING
            elif score >= lost_thr:
                track.state = STATE_SUSPICIOUS
            else:
                track.state = STATE_LOST

        # 2. Lost Count Tracking
        if mask is not None and track.state in (STATE_RELIABLE, STATE_PENDING):
            track.lost_count = 0
        else:
            track.lost_count += 1

        # 3. Termination
        if track.lost_count >= lost_tolerance:
            track.active = False

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

    def add_new_objects(
        self,
        frame_id: int,
        frame_idx: int,
        detections: List[Detection],
        wrapper,
    ) -> List[Track]:
        """
        Dynamically add new objects appearing mid-sequence.
        - Logic:
          1. Filter detections with score >= det_conf_thr (0.5).
          2. Compute IoU matching (iou_match_thr=0.5) between active tracks and detections.
          3. Treat unmatched detections as candidates.
          4. Compute free_mask = not union_masks([t.mask for t in active_tracks]).
          5. Compute free_area_ratio for candidate boxes.
          6. If free_area_ratio >= free_ratio_thr (0.7), initialize as a new track with a SAM2 box prompt.
        """
        enable_addition = self.config.get("enable_object_addition", True)
        if not enable_addition:
            return []

        det_conf_thr = self.config.get("det_conf_thr", 0.5)
        iou_match_thr = self.config.get("iou_match_thr", 0.5)
        free_ratio_thr = self.config.get("free_ratio_thr", 0.7)

        # 1. Filter detections with score >= det_conf_thr
        filtered_dets = [d for d in detections if d.score >= det_conf_thr]
        if not filtered_dets:
            return []

        active_tracks = self.get_active_tracks()

        # 2. Compute IoU matching between active tracks and detections
        track_boxes = [t.bbox for t in active_tracks if t.bbox is not None]
        det_boxes = [d.box_xyxy for d in filtered_dets]

        matches, unmatched_tracks, unmatched_detections = iou_matching(
            track_boxes, det_boxes, iou_thr=iou_match_thr
        )

        # 3. Treat unmatched detections as candidates
        candidates = [filtered_dets[i] for i in unmatched_detections]
        if not candidates:
            return []

        # 4. Compute free_mask = not union_masks([t.mask for t in active_tracks])
        # Find H, W from existing masks or wrapper inference state if no masks exist yet
        H, W = None, None
        for t in active_tracks:
            if t.mask is not None:
                H, W = t.mask.shape
                break

        if H is None or W is None:
            # Fall back to wrapper dimensions
            if hasattr(wrapper, "inference_state") and wrapper.inference_state is not None:
                H = wrapper.inference_state.get("video_height")
                W = wrapper.inference_state.get("video_width")
            # If still None, default to a standard size
            if H is None or W is None:
                H, W = 480, 640

        valid_masks = [t.mask for t in active_tracks if t.mask is not None]
        union = union_masks(valid_masks)
        if union is not None:
            free_mask = np.logical_not(union)
        else:
            free_mask = np.ones((H, W), dtype=bool)

        newly_added: List[Track] = []

        # 5. Compute free_area_ratio and add new tracks
        for det in candidates:
            ratio = compute_free_area_ratio(det.box_xyxy, free_mask)
            if ratio >= free_ratio_thr:
                # 6. Initialize as a new track with a SAM2 box prompt
                obj_id = self.next_track_id  # Create new unique ID
                mask, bbox, score = wrapper.add_box_prompt(
                    frame_idx, obj_id, det.box_xyxy
                )
                if mask is not None:
                    track = self.create_track(
                        frame_id=frame_id,
                        mask=mask,
                        bbox=bbox,
                        score=score,
                        keyframe_idx=frame_idx,
                    )
                    newly_added.append(track)
                    # Update free_mask so subsequent candidates in the same frame don't overlap
                    free_mask = np.logical_and(free_mask, np.logical_not(mask))

        return newly_added

    def associate_and_update(
        self,
        frame_id: int,
        frame_idx: int,
        detections: List[Detection],
        wrapper,
    ) -> bool:
        """
        Perform association, quality reconstruction (re-prompting pending tracks),
        and object addition (prompting new tracks) in a single unified step.
        
        Returns:
            triggered_restart: bool, True if any track was prompted or re-prompted, 
                              indicating that propagation should break and restart.
        """
        det_conf_thr = self.config.get("det_conf_thr", 0.5)
        iou_match_thr = self.config.get("iou_match_thr", 0.5)
        free_ratio_thr = self.config.get("free_ratio_thr", 0.7)
        enable_reconstruction = self.config.get("enable_quality_reconstruction", True)
        enable_addition = self.config.get("enable_object_addition", True)

        # 1. Filter detections with score >= det_conf_thr
        filtered_dets = [d for d in detections if d.score >= det_conf_thr]
        if not filtered_dets:
            return False

        active_tracks = self.get_active_tracks()

        # 2. Compute IoU matching between active tracks and detections
        track_boxes = [t.bbox for t in active_tracks if t.bbox is not None]
        det_boxes = [d.box_xyxy for d in filtered_dets]

        matches, unmatched_tracks, unmatched_detections = iou_matching(
            track_boxes, det_boxes, iou_thr=iou_match_thr
        )

        triggered_restart = False

        # 3. Quality Reconstruction for matched pending tracks
        if enable_reconstruction:
            from tracker.track import STATE_PENDING
            for t_idx, d_idx in matches:
                track = active_tracks[t_idx]
                det = filtered_dets[d_idx]
                if track.state == STATE_PENDING and track.keyframe_idx != frame_idx:
                    print(
                        f"[TrajectoryManager] Re-prompting Track {track.track_id} (state: {track.state}) "
                        f"using detection at frame {frame_id} for Quality Reconstruction."
                    )
                    mask, bbox, score = wrapper.add_box_prompt(
                        frame_idx, track.track_id, det.box_xyxy
                    )
                    if mask is not None:
                        # Update track keyframe index
                        track.keyframe_idx = frame_idx
                        triggered_restart = True

        # 4. Object Addition for unmatched detections
        if enable_addition:
            # Find H, W from existing masks or wrapper inference state if no masks exist yet
            H, W = None, None
            for t in active_tracks:
                if t.mask is not None:
                    H, W = t.mask.shape
                    break

            if H is None or W is None:
                # Fall back to wrapper dimensions
                if hasattr(wrapper, "inference_state") and wrapper.inference_state is not None:
                    H = wrapper.inference_state.get("video_height")
                    W = wrapper.inference_state.get("video_width")
                if H is None or W is None:
                    H, W = 480, 640

            valid_masks = [t.mask for t in active_tracks if t.mask is not None]
            union = union_masks(valid_masks)
            if union is not None:
                free_mask = np.logical_not(union)
            else:
                free_mask = np.ones((H, W), dtype=bool)

            candidates = [filtered_dets[i] for i in unmatched_detections]
            for det in candidates:
                ratio = compute_free_area_ratio(det.box_xyxy, free_mask)
                if ratio >= free_ratio_thr:
                    obj_id = self.next_track_id
                    mask, bbox, score = wrapper.add_box_prompt(
                        frame_idx, obj_id, det.box_xyxy
                    )
                    if mask is not None:
                        track = self.create_track(
                            frame_id=frame_id,
                            mask=mask,
                            bbox=bbox,
                            score=score,
                            keyframe_idx=frame_idx,
                        )
                        # Update free_mask so subsequent candidates in this frame don't overlap
                        free_mask = np.logical_and(free_mask, np.logical_not(mask))
                        triggered_restart = True

        return triggered_restart

    def resolve_conflicts(self) -> None:
        """
        Milestone 8: Cross-object Interaction Approximation.
        Detect conflicts where active tracks overlap significantly in their binary masks,
        and degrade/corrupt the less stable track.
        """
        enable_coi = self.config.get("enable_cross_object_interaction", False)
        if not enable_coi:
            return

        coi_miou_thr = self.config.get("coi_miou_thr", 0.8)
        coi_score_gap_thr = self.config.get("coi_score_gap_thr", 2.0)
        coi_var_window = self.config.get("coi_var_window", 10)

        active_tracks = self.get_active_tracks()
        # Only check active tracks with a valid mask
        valid_tracks = [t for t in active_tracks if t.mask is not None]

        corrupted_track_ids = set()

        for i in range(len(valid_tracks)):
            for j in range(i + 1, len(valid_tracks)):
                ti = valid_tracks[i]
                tj = valid_tracks[j]

                if ti.track_id in corrupted_track_ids or tj.track_id in corrupted_track_ids:
                    continue

                miou = mask_iou(ti.mask, tj.mask)
                if miou > coi_miou_thr:
                    # Conflict found!
                    score_i = ti.score
                    score_j = tj.score

                    # 1. Score Gap Check
                    if abs(score_i - score_j) > coi_score_gap_thr:
                        if score_i < score_j:
                            corrupted_track_ids.add(ti.track_id)
                        else:
                            corrupted_track_ids.add(tj.track_id)
                    else:
                        # 2. Variance Check
                        hist_i = ti.score_history[-coi_var_window:]
                        hist_j = tj.score_history[-coi_var_window:]

                        var_i = float(np.var(hist_i)) if hist_i else 0.0
                        var_j = float(np.var(hist_j)) if hist_j else 0.0

                        if abs(var_i - var_j) > 1e-9:
                            if var_i > var_j:
                                corrupted_track_ids.add(ti.track_id)
                            else:
                                corrupted_track_ids.add(tj.track_id)
                        else:
                            # 3. Tie-breaker
                            if score_i < score_j:
                                corrupted_track_ids.add(ti.track_id)
                            else:
                                corrupted_track_ids.add(tj.track_id)

        # Degrade corrupted tracks
        for track in valid_tracks:
            if track.track_id in corrupted_track_ids:
                print(f"[TrajectoryManager] Track {track.track_id} marked as corrupted due to mask conflict.")
                track.corrupted = True
                track.state = STATE_SUSPICIOUS
                track.mask = None
                track.bbox = None
