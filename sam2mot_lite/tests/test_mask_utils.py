import unittest
import numpy as np
from tracker.mask_utils import (
    mask_to_box, mask_iou, union_masks, bbox_iou, compute_free_area_ratio
)

class TestMaskUtils(unittest.TestCase):
    def test_mask_to_box(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        self.assertIsNone(mask_to_box(mask))
        
        mask[10:20, 30:40] = 1
        box = mask_to_box(mask)
        self.assertEqual(box, [30.0, 10.0, 39.0, 19.0])

    def test_mask_iou(self):
        mask_a = np.zeros((10, 10), dtype=np.uint8)
        mask_b = np.zeros((10, 10), dtype=np.uint8)
        
        self.assertEqual(mask_iou(mask_a, mask_b), 0.0)
        
        mask_a[2:5, 2:5] = 1
        mask_b[2:5, 2:5] = 1
        self.assertEqual(mask_iou(mask_a, mask_b), 1.0)
        
        mask_b[:] = 0
        mask_b[4:7, 4:7] = 1
        # Intersection: 4:5, 4:5 -> 1 cell
        # Area A: 9, Area B: 9 -> Union: 17
        self.assertAlmostEqual(mask_iou(mask_a, mask_b), 1.0 / 17.0)

    def test_union_masks(self):
        self.assertIsNone(union_masks([]))
        
        m1 = np.zeros((10, 10), dtype=bool)
        m1[0:2, 0:2] = True
        m2 = np.zeros((10, 10), dtype=bool)
        m2[1:3, 1:3] = True
        
        union = union_masks([m1, m2])
        self.assertTrue(union[0, 0])
        self.assertTrue(union[2, 2])
        self.assertFalse(union[9, 9])
        self.assertEqual(union.sum(), 7)

    def test_bbox_iou(self):
        box_a = [0, 0, 10, 10]
        box_b = [0, 0, 10, 10]
        self.assertEqual(bbox_iou(box_a, box_b), 1.0)
        
        box_c = [10, 10, 20, 20]
        self.assertEqual(bbox_iou(box_a, box_c), 0.0)
        
        box_d = [5, 5, 15, 15]
        # Inter: 5,5 -> 10,10 = 25
        # Area A: 100, Area B: 100, Union: 175
        self.assertAlmostEqual(bbox_iou(box_a, box_d), 25.0 / 175.0)

    def test_compute_free_area_ratio(self):
        free_mask = np.ones((100, 100), dtype=np.uint8)
        box = [10.0, 10.0, 20.0, 20.0]
        
        ratio = compute_free_area_ratio(box, free_mask)
        self.assertEqual(ratio, 1.0)
        
        free_mask[10:15, 10:20] = 0
        ratio2 = compute_free_area_ratio(box, free_mask)
        self.assertEqual(ratio2, 0.5)

if __name__ == '__main__':
    unittest.main()
