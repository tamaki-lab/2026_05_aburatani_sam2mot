import unittest
import os
import tempfile
from tracker.result_writer import write_trajectories

class TestResultWriter(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.out_file = os.path.join(self.temp_dir.name, "trajectories.txt")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_write_trajectories(self):
        trajectories = [
            {'frame_id': 1, 'track_id': 1, 'bbox_xywh': [10.5, 20.5, 30.5, 40.5], 'score': 0.95},
            {'frame_id': 2, 'track_id': 1, 'bbox_xywh': [15.0, 25.0, 30.0, 40.0], 'score': 0.90},
            {'frame_id': 2, 'track_id': 2, 'bbox_xywh': [100.0, 200.0, 50.0, 50.0], 'score': 0.85},
        ]
        
        write_trajectories(self.out_file, trajectories)
        
        self.assertTrue(os.path.exists(self.out_file))
        
        with open(self.out_file, 'r') as f:
            lines = f.readlines()
            
        self.assertEqual(len(lines), 3)
        self.assertEqual(lines[0].strip(), "1,1,10.50,20.50,30.50,40.50,0.9500,-1,-1,-1")
        self.assertEqual(lines[1].strip(), "2,1,15.00,25.00,30.00,40.00,0.9000,-1,-1,-1")
        self.assertEqual(lines[2].strip(), "2,2,100.00,200.00,50.00,50.00,0.8500,-1,-1,-1")

if __name__ == '__main__':
    unittest.main()
