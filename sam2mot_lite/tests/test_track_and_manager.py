"""
Unit tests for Track and TrajectoryManager (CPU only, SAM2 不要).
"""
import unittest
import numpy as np

from tracker.track import (
    Track,
    STATE_RELIABLE,
    STATE_PENDING,
    STATE_SUSPICIOUS,
    STATE_LOST,
)
from tracker.trajectory_manager import TrajectoryManager


class TestTrack(unittest.TestCase):
    def test_default_values(self):
        t = Track(track_id=1)
        self.assertEqual(t.track_id, 1)
        self.assertEqual(t.state, STATE_RELIABLE)
        self.assertTrue(t.active)
        self.assertIsNone(t.mask)
        self.assertIsNone(t.bbox)
        self.assertEqual(t.score, 0.0)
        self.assertEqual(t.score_history, [])
        self.assertEqual(t.lost_count, 0)
        self.assertEqual(t.last_frame, 0)
        self.assertEqual(t.keyframe_idx, 0)

    def test_state_constants(self):
        self.assertEqual(STATE_RELIABLE, "reliable")
        self.assertEqual(STATE_PENDING, "pending")
        self.assertEqual(STATE_SUSPICIOUS, "suspicious")
        self.assertEqual(STATE_LOST, "lost")


class TestTrajectoryManager(unittest.TestCase):
    def setUp(self):
        self.config = {"det_conf_thr": 0.5}
        self.tm = TrajectoryManager(self.config)

    def test_create_track_increments_id(self):
        t1 = self.tm.create_track(
            frame_id=1, mask=None, bbox=[0, 0, 10, 10], score=0.9, keyframe_idx=0
        )
        t2 = self.tm.create_track(
            frame_id=1, mask=None, bbox=[20, 20, 30, 30], score=0.8, keyframe_idx=0
        )
        self.assertEqual(t1.track_id, 1)
        self.assertEqual(t2.track_id, 2)
        self.assertEqual(self.tm.next_track_id, 3)
        self.assertEqual(len(self.tm.tracks), 2)

    def test_get_active_tracks(self):
        t1 = self.tm.create_track(1, None, [0, 0, 10, 10], 0.9, 0)
        t2 = self.tm.create_track(1, None, [20, 20, 30, 30], 0.8, 0)
        t2.active = False

        active = self.tm.get_active_tracks()
        self.assertEqual(len(active), 1)
        self.assertEqual(active[0].track_id, 1)

    def test_update_track_with_mask(self):
        t = self.tm.create_track(1, None, [0, 0, 10, 10], 0.9, 0)
        mask = np.ones((10, 10), dtype=np.uint8)

        self.tm.update_track(t, frame_id=2, mask=mask, bbox=[1, 1, 11, 11], score=0.85)

        self.assertEqual(t.last_frame, 2)
        self.assertEqual(t.score, 0.85)
        self.assertEqual(len(t.score_history), 2)
        self.assertEqual(t.lost_count, 0)
        self.assertIsNotNone(t.mask)

    def test_update_track_without_mask_increments_lost(self):
        t = self.tm.create_track(1, None, [0, 0, 10, 10], 0.9, 0)

        self.tm.update_track(t, frame_id=2, mask=None, bbox=None, score=0.0)
        self.assertEqual(t.lost_count, 1)

        self.tm.update_track(t, frame_id=3, mask=None, bbox=None, score=0.0)
        self.assertEqual(t.lost_count, 2)

    def test_get_track_by_id(self):
        self.tm.create_track(1, None, [0, 0, 10, 10], 0.9, 0)
        self.tm.create_track(1, None, [20, 20, 30, 30], 0.8, 0)

        t = self.tm.get_track_by_id(2)
        self.assertIsNotNone(t)
        self.assertEqual(t.track_id, 2)

        self.assertIsNone(self.tm.get_track_by_id(999))

    def test_score_history_accumulates(self):
        t = self.tm.create_track(1, None, [0, 0, 10, 10], 0.9, 0)
        self.tm.update_track(t, 2, None, None, 0.7)
        self.tm.update_track(t, 3, None, None, 0.5)
        self.assertEqual(t.score_history, [0.9, 0.7, 0.5])

    def test_add_new_objects_no_addition_config(self):
        # When addition is disabled, it should return empty
        self.tm.config["enable_object_addition"] = False
        from tracker.detection import Detection
        dets = [Detection(frame_id=2, box_xyxy=[0, 0, 10, 10], score=0.9)]
        res = self.tm.add_new_objects(frame_id=2, frame_idx=1, detections=dets, wrapper=None)
        self.assertEqual(res, [])

    def test_add_new_objects_below_conf_threshold(self):
        from tracker.detection import Detection
        dets = [Detection(frame_id=2, box_xyxy=[0, 0, 10, 10], score=0.4)]
        res = self.tm.add_new_objects(frame_id=2, frame_idx=1, detections=dets, wrapper=None)
        self.assertEqual(res, [])

    def test_add_new_objects_matching_existing_active_track(self):
        from tracker.detection import Detection
        # Existing track has bbox [0, 0, 10, 10]
        self.tm.create_track(1, None, [0, 0, 10, 10], 0.9, 0)
        
        # Detection matches the existing track (IoU = 1.0)
        dets = [Detection(frame_id=2, box_xyxy=[0, 0, 10, 10], score=0.9)]
        res = self.tm.add_new_objects(frame_id=2, frame_idx=1, detections=dets, wrapper=None)
        self.assertEqual(res, [])

    def test_add_new_objects_low_free_area_ratio(self):
        from tracker.detection import Detection
        # Existing track has a mask at [0, 0, 100, 100]
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[0:50, 0:50] = 1
        t = self.tm.create_track(1, mask, [0, 0, 50, 50], 0.9, 0)
        
        # Detection overlaps heavily with the existing mask (free area ratio is low)
        # Bbox is [10, 10, 30, 30] which is entirely inside the mask, so free area ratio is 0.0
        dets = [Detection(frame_id=2, box_xyxy=[10, 10, 30, 30], score=0.9)]
        res = self.tm.add_new_objects(frame_id=2, frame_idx=1, detections=dets, wrapper=None)
        self.assertEqual(res, [])

    def test_add_new_objects_successful_addition(self):
        from tracker.detection import Detection
        # Existing track has a mask at [0, 0, 50, 50]
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[0:50, 0:50] = 1
        t = self.tm.create_track(1, mask, [0, 0, 50, 50], 0.9, 0)
        
        # Detection is in the free area: [60, 60, 90, 90]
        dets = [Detection(frame_id=2, box_xyxy=[60, 60, 90, 90], score=0.9)]
        
        # Mock SAM2 wrapper
        class MockWrapper:
            def __init__(self):
                self.inference_state = {"video_height": 100, "video_width": 100}
            def add_box_prompt(self, frame_idx, obj_id, box_xyxy):
                # Return a valid dummy mask
                m = np.zeros((100, 100), dtype=np.uint8)
                m[60:90, 60:90] = 1
                return m, box_xyxy, 0.95
                
        wrapper = MockWrapper()
        res = self.tm.add_new_objects(frame_id=2, frame_idx=1, detections=dets, wrapper=wrapper)
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].track_id, 2)
        self.assertEqual(res[0].bbox, [60, 60, 90, 90])


if __name__ == "__main__":
    unittest.main()
