# SAM2MOT-lite

A modular Tracking-by-Segmentation baseline applying Segment Anything Model 2 (SAM2) to Multi-Object Tracking (MOT), inspired by the SAM2MOT architecture.

> [!WARNING]
> Due to execution environment issues (Exit Code 255), the test scripts below have NOT been run and verified dynamically yet. Please run them locally to confirm functionality.

## Current Status (Milestones 0 - 3)
We have implemented the following components focusing purely on SAM2 API wrapper, detection reading, and mask conversion. 
- **Note:** Frame-by-frame MOT inference loop, Object Addition, and Object Removal are **NOT** implemented yet in M0-M3. We only test adding prompts and saving masks using the SAM2 API.
- `sam2/` core logic and checkpoints remain unmodified.
- Code resides in `sam2mot_lite/`.

## Input/Output Format

**Input (detections.txt)**: Supports standard MOT format (`frame, id, x, y, w, h, score, class, visibility`) and minimal format (`frame, x, y, w, h, score`). *Note: ID column is ignored and not used as track_id during M0.*

**Output (trajectories.txt)**: TrackEval compatible format (`frame, track_id, x, y, w, h, score, -1, -1, -1`). Bounding boxes are strictly derived from SAM2 masks.

*(Note: MOT uses 1-indexed frames, while SAM2 uses 0-indexed frames. Explicit conversion functions are used internally).*

## Testing & Execution

### 1. Unit Tests (CPU/General)
These tests verify core utility functions without loading SAM2 models.

```bash
cd /path/to/sam2
PYTHONPATH=./sam2mot_lite python -m unittest discover sam2mot_lite/tests/
```

### 2. Smoke Tests (GPU & SAM2 Checkpoint Required)
These scripts load the SAM2 model and verify mask generation from dummy video frames.

**Arguments for Smoke Tests:**
- `--config`: Path to SAM2 yaml config (default: `sam2/configs/sam2.1/sam2.1_hiera_t.yaml`)
- `--checkpoint`: Path to SAM2 checkpoint (default: `sam2/checkpoints/sam2.1_hiera_tiny.pt`)

**Single Object Smoke Test:**
```bash
PYTHONPATH=./sam2mot_lite python sam2mot_lite/tests/smoke_sam2_single_object.py
```
*Expected:* Initializes a dummy video, prompts for an object, and saves `mask_0.npz`.

**Multi Object Smoke Test:**
```bash
PYTHONPATH=./sam2mot_lite python sam2mot_lite/tests/smoke_sam2_multi_object.py
```
*Expected:* Adds prompts for 2 objects, propagates across 5 frames, and saves results to `trajectories.txt`.
