"""
Unit tests untuk core/sdi.py
Menguji batas-batas perhitungan SDI sesuai Manual PKRMS Tabel 9 & 10.
"""
import unittest
from core.sdi import (
    compute_sdi_level_1,
    compute_sdi_level_2,
    compute_sdi_level_3,
    compute_sdi_level_4,
    classify_sdi_condition,
    calculate_sdi,
)


class TestSDICalculation(unittest.TestCase):

    def test_sdi_level_1_boundaries(self):
        # 0% -> 0
        self.assertEqual(compute_sdi_level_1(0.0), 0.0)
        self.assertEqual(compute_sdi_level_1(-1.0), 0.0)

        # < 10% -> 5
        self.assertEqual(compute_sdi_level_1(0.1), 5.0)
        self.assertEqual(compute_sdi_level_1(5.0), 5.0)
        self.assertEqual(compute_sdi_level_1(9.99), 5.0)

        # 10 - 30% -> 20
        self.assertEqual(compute_sdi_level_1(10.0), 20.0)
        self.assertEqual(compute_sdi_level_1(20.0), 20.0)
        self.assertEqual(compute_sdi_level_1(30.0), 20.0)

        # > 30% -> 40
        self.assertEqual(compute_sdi_level_1(30.01), 40.0)
        self.assertEqual(compute_sdi_level_1(50.0), 40.0)

    def test_sdi_level_2_boundaries(self):
        sdi1 = 20.0
        # <= 3 mm -> SDI1
        self.assertEqual(compute_sdi_level_2(sdi1, 0.0, crack_width_threshold_mm=3.0), 20.0)
        self.assertEqual(compute_sdi_level_2(sdi1, 1.0, crack_width_threshold_mm=3.0), 20.0)
        self.assertEqual(compute_sdi_level_2(sdi1, 3.0, crack_width_threshold_mm=3.0), 20.0)

        # > 3 mm -> SDI1 * 2
        self.assertEqual(compute_sdi_level_2(sdi1, 3.01, crack_width_threshold_mm=3.0), 40.0)
        self.assertEqual(compute_sdi_level_2(sdi1, 5.0, crack_width_threshold_mm=3.0), 40.0)

        # Uji threshold kustom misal 5.0 mm (narasi teks hal 38)
        self.assertEqual(compute_sdi_level_2(sdi1, 4.0, crack_width_threshold_mm=5.0), 20.0)
        self.assertEqual(compute_sdi_level_2(sdi1, 5.01, crack_width_threshold_mm=5.0), 40.0)

    def test_sdi_level_3_boundaries(self):
        sdi2 = 20.0
        # None -> SDI2
        self.assertEqual(compute_sdi_level_3(sdi2, 0.0), 20.0)
        self.assertEqual(compute_sdi_level_3(sdi2, -5.0), 20.0)

        # < 10 per km -> +15
        self.assertEqual(compute_sdi_level_3(sdi2, 0.1), 35.0)
        self.assertEqual(compute_sdi_level_3(sdi2, 9.9), 35.0)

        # 10 - 50 per km -> +75
        self.assertEqual(compute_sdi_level_3(sdi2, 10.0), 95.0)
        self.assertEqual(compute_sdi_level_3(sdi2, 30.0), 95.0)
        self.assertEqual(compute_sdi_level_3(sdi2, 50.0), 95.0)

        # > 50 per km -> +225
        self.assertEqual(compute_sdi_level_3(sdi2, 50.1), 245.0)
        self.assertEqual(compute_sdi_level_3(sdi2, 80.0), 245.0)

    def test_sdi_level_4_boundaries(self):
        sdi3 = 50.0
        # None (0 cm) -> SDI3
        self.assertEqual(compute_sdi_level_4(sdi3, 0.0), 50.0)
        self.assertEqual(compute_sdi_level_4(sdi3, -1.0), 50.0)

        # < 1 cm -> +2.5
        self.assertEqual(compute_sdi_level_4(sdi3, 0.5), 52.5)
        self.assertEqual(compute_sdi_level_4(sdi3, 0.99), 52.5)

        # 1 - 3 cm -> +10.0
        self.assertEqual(compute_sdi_level_4(sdi3, 1.0), 60.0)
        self.assertEqual(compute_sdi_level_4(sdi3, 2.0), 60.0)
        self.assertEqual(compute_sdi_level_4(sdi3, 3.0), 60.0)

        # > 3 cm -> +20.0
        self.assertEqual(compute_sdi_level_4(sdi3, 3.01), 70.0)
        self.assertEqual(compute_sdi_level_4(sdi3, 5.0), 70.0)

    def test_classify_sdi_condition(self):
        self.assertEqual(classify_sdi_condition(0.0)["label"], "Baik")
        self.assertEqual(classify_sdi_condition(49.9)["label"], "Baik")
        self.assertEqual(classify_sdi_condition(50.0)["label"], "Sedang")
        self.assertEqual(classify_sdi_condition(99.9)["label"], "Sedang")
        self.assertEqual(classify_sdi_condition(100.0)["label"], "Rusak Ringan")
        self.assertEqual(classify_sdi_condition(149.9)["label"], "Rusak Ringan")
        self.assertEqual(classify_sdi_condition(150.0)["label"], "Rusak Berat")
        self.assertEqual(classify_sdi_condition(300.0)["label"], "Rusak Berat")

    def test_calculate_sdi_full_pipeline(self):
        res = calculate_sdi(
            crack_area_pct=15.0,
            avg_crack_width_mm=4.0,
            potholes_per_km=12.0,
            rut_depth_cm=2.0,
            crack_width_threshold_mm=3.0
        )
        self.assertEqual(res.sdi1, 20.0)
        self.assertEqual(res.sdi2, 40.0)
        self.assertEqual(res.sdi3, 115.0)
        self.assertEqual(res.sdi4, 125.0)
        self.assertEqual(res.final_sdi, 125.0)
        self.assertEqual(res.condition, "Rusak Ringan")


if __name__ == "__main__":
    unittest.main()
