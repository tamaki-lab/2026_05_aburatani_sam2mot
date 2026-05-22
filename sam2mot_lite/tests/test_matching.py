import unittest
from tracker.matching import iou_matching

class TestMatching(unittest.TestCase):
    def test_iou_matching(self):
        # tracks: [0, 0, 10, 10], [20, 20, 30, 30]
        tracks = [[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]]
        
        # det1 matches track0 perfectly. det2 matches track1 partially. det3 is unmatched.
        detections = [
            [0.0, 0.0, 10.0, 10.0],
            [22.0, 22.0, 32.0, 32.0],
            [50.0, 50.0, 60.0, 60.0]
        ]
        
        matches, un_tracks, un_dets = iou_matching(tracks, detections, iou_thr=0.1)
        
        self.assertIn((0, 0), matches)
        self.assertIn((1, 1), matches)
        self.assertEqual(len(matches), 2)
        
        self.assertEqual(un_tracks, [])
        self.assertEqual(un_dets, [2])

    def test_no_matches(self):
        tracks = [[0.0, 0.0, 10.0, 10.0]]
        detections = [[20.0, 20.0, 30.0, 30.0]]
        
        matches, un_tracks, un_dets = iou_matching(tracks, detections, iou_thr=0.1)
        self.assertEqual(matches, [])
        self.assertEqual(un_tracks, [0])
        self.assertEqual(un_dets, [0])

if __name__ == '__main__':
    unittest.main()
