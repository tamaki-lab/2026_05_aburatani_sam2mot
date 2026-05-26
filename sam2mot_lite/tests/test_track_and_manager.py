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


if __name__ == "__main__":
    unittest.main()
