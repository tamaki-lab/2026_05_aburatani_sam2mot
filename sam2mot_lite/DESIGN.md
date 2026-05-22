# SAM2MOT-lite Design Document (Milestones 4-9)

This document outlines the architectural and functional design for the remaining milestones of the SAM2MOT-lite implementation, transitioning from a static multi-object wrapper to a fully-fledged Tracking-by-Segmentation pipeline.

## Milestone 4: Minimal SAM2MOT-lite Inference
**Goal**: Implement a complete tracker utilizing SAM2 prompts for initialization.
- **`Track` dataclass**: Maintains track state (`track_id`, `state`, `mask`, `bbox`, `score`, `score_history`, `lost_count`, `last_frame`, `keyframe_idx`, `active`).
- **`TrajectoryManager`**: Manages the lifecycle of tracks.
- **Workflow**: For each frame, read high-confidence detections, initialize new tracks via SAM2 box prompts, propagate masks to the next frame, update `Track` states, and write bounding boxes derived from masks to `trajectories.txt`.

## Milestone 5: Object Addition
**Goal**: Dynamically add new objects appearing mid-sequence.
- **Logic**:
  1. Filter detections with score `>= det_conf_thr` (0.5).
  2. Compute IoU matching (`iou_match_thr=0.5`) between active tracks and detections.
  3. Treat unmatched detections as candidates.
  4. Compute `free_mask = not union_masks([t.mask for t in active_tracks])`.
  5. Compute `free_area_ratio` for candidate boxes.
  6. If `free_area_ratio >= free_ratio_thr` (0.7), initialize as a new track with a SAM2 box prompt.

## Milestone 6: Object Removal
**Goal**: Manage track states and terminate lost tracks.
- **States**: `reliable`, `pending`, `suspicious`, `lost`.
- **Logic**: Update state based on SAM2 confidence scores (provisional scoring). If score drops below thresholds (`pending_thr=0.0`, `lost_thr=-2.0`), degrade state.
- **Termination**: If a track is `lost` or `suspicious` for `lost_tolerance` (25) consecutive frames, set `active=False`. Only output active/pending/reliable tracks.

## Milestone 7: Quality Reconstruction
**Goal**: Recover degrading tracks using new detection prompts.
- **Logic**: For tracks in `pending` state, if an IoU match exists with a high-confidence detection, re-prompt SAM2 for the same `obj_id` using the matched detection box.
- Update `keyframe_idx` and log the reconstruction event. Configurable via `enable_quality_reconstruction`.

## Milestone 8: Cross-object Interaction Approximation
**Goal**: Handle identity switches and mask overlapping conflicts without deep SAM2 memory modification.
- **Logic**: Calculate pairwise mask IoU for active tracks. If mIoU > `coi_miou_thr` (0.8), a conflict exists.
- **Resolution**: Track with a significantly lower score (`coi_score_gap_thr=2.0`) is marked `corrupted`. If scores are similar, use historical variance over `coi_var_window` (10). Suppress output for `corrupted` tracks and degrade their state to `suspicious`.

## Milestone 9: TrackEval Integration
**Goal**: Evaluate tracking performance seamlessly using the TrackEval library.
- Format `trajectories.txt` strictly to MOT Challenge specifications (1-indexed frames and IDs, xywh format).
- Create a streamlined script (`scripts/run_sequence.py`) accepting standard directories, outputting directly to TrackEval-compatible folders.
- Document comparison workflows across different configuration toggles (e.g., Object Addition ON vs OFF).
