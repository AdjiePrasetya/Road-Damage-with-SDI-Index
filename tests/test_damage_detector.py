import unittest
import numpy as np
from models.damage_detector import RoadDamageDetector


class TestRoadDamageDetector(unittest.TestCase):
    def setUp(self):
        self.detector = RoadDamageDetector(
            meter_per_pixel=0.005
        )

    def test_detector_initialization(self):
        """Memverifikasi bahwa 3 model segmentasi dapat dimuat jika bobot tersedia."""
        self.assertIsNotNone(self.detector.class_mapping)
        self.assertIn("crack_gt_3mm", self.detector.class_mapping)
        self.assertIn("pothole", self.detector.class_mapping)
        self.assertIn("rut_gt_3cm", self.detector.class_mapping)

    def test_empty_frame_prediction(self):
        """Frame dummy kosong tidak menghasilkan error."""
        dummy_frame = np.zeros((360, 640, 3), dtype=np.uint8)
        results = self.detector.predict(dummy_frame)
        self.assertIsInstance(results, list)


if __name__ == "__main__":
    unittest.main()

