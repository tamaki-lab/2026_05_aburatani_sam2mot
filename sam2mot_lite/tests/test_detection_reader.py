import unittest
import os
import tempfile
from tracker.detection import Detection, read_detections, filter_detections_by_score

class TestDetectionReader(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.det_file = os.path.join(self.temp_dir.name, "detections.txt")
        with open(self.det_file, "w") as f:
            # frame, id, x, y, w, h, score, class, visibility
            f.write("1,-1,10.0,20.0,30.0,40.0,0.9,1,1.0\n")
            f.write("1,-1,100.0,200.0,50.0,50.0,0.3,1,1.0\n")
            f.write("2,-1,15.0,25.0,30.0,40.0,0.85,1,1.0\n")
            f.write("invalid,line\n")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_xywh_to_xyxy(self):
        det = Detection.from_xywh(1, 10, 20, 30, 40, 0.9)
        self.assertEqual(det.box_xyxy, [10, 20, 40, 60])
        self.assertEqual(det.to_xywh(), [10, 20, 30, 40])

    def test_read_detections(self):
        dets = read_detections(self.det_file)
        self.assertEqual(len(dets), 3)
        self.assertEqual(dets[0].frame_id, 1)
        self.assertEqual(dets[0].box_xyxy, [10.0, 20.0, 40.0, 60.0])
        self.assertEqual(dets[0].score, 0.9)
        self.assertEqual(dets[0].cls, 1)

    def test_read_with_threshold(self):
        dets = read_detections(self.det_file, score_thr=0.5)
        self.assertEqual(len(dets), 2)
        self.assertEqual(dets[0].score, 0.9)
        self.assertEqual(dets[1].score, 0.85)

    def test_filter_by_score(self):
        dets = read_detections(self.det_file)
        filtered = filter_detections_by_score(dets, 0.5)
        self.assertEqual(len(filtered), 2)

if __name__ == '__main__':
    unittest.main()
