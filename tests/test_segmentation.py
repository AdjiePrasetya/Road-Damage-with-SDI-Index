"""
Unit tests untuk core/segmentation.py
Menguji pembagian blok 100m, pemotongan segmen saat perkerasan berubah,
penggabungan segmen sangat pendek (<10m), dan perbaikan manual.
"""
import unittest
from core.segmentation import (
    FrameRecord,
    create_segments_from_frames,
    smooth_pavement_types,
    apply_manual_corrections
)


class TestSegmentation(unittest.TestCase):

    def test_smoothing_pavement_types(self):
        frames = [
            FrameRecord(frame_idx=0, sta_m=0.0, pavement_type="hotmix", pavement_conf=0.9),
            FrameRecord(frame_idx=1, sta_m=10.0, pavement_type="hotmix", pavement_conf=0.9),
            FrameRecord(frame_idx=2, sta_m=20.0, pavement_type="perkerasan_beton", pavement_conf=0.5), # flicker noise
            FrameRecord(frame_idx=3, sta_m=30.0, pavement_type="hotmix", pavement_conf=0.9),
            FrameRecord(frame_idx=4, sta_m=40.0, pavement_type="hotmix", pavement_conf=0.9),
        ]
        smoothed = smooth_pavement_types(frames, window_size=3, min_run_length=2)
        self.assertEqual(smoothed[2], "hotmix")
        self.assertTrue(all(label == "hotmix" for label in smoothed))

    def test_segment_split_on_pavement_change(self):
        """
        Skenario sesuai contoh spesifikasi pengguna:
        Jalan Ilyas Bedul | 0 - 60 m Hotmix | 60 - 100 m Beton
        """
        frames = []
        # 0 s.d 55 m: Hotmix
        for sta in range(0, 60, 5):
            frames.append(FrameRecord(frame_idx=len(frames), sta_m=float(sta), pavement_type="Hotmix", pavement_conf=0.95))
        # 60 s.d 100 m: Beton
        for sta in range(60, 105, 5):
            frames.append(FrameRecord(frame_idx=len(frames), sta_m=float(sta), pavement_type="Beton", pavement_conf=0.95))

        segments = create_segments_from_frames(
            frames=frames,
            road_name="Jalan Ilyas Bedul",
            sta_start_m=0.0,
            base_segment_length_m=100.0,
            min_segment_length_m=10.0,
            smoothing_window=3,
            min_run_length=2
        )

        self.assertEqual(len(segments), 2)
        # Segmen 1: 0 - 60 Hotmix
        self.assertEqual(segments[0].sta_start, 0.0)
        self.assertEqual(segments[0].sta_end, 60.0)
        self.assertEqual(segments[0].length_m, 60.0)
        self.assertEqual(segments[0].pavement_type, "Hotmix")

        # Segmen 2: 60 - 100 Beton
        self.assertEqual(segments[1].sta_start, 60.0)
        self.assertEqual(segments[1].sta_end, 100.0)
        self.assertEqual(segments[1].length_m, 40.0)
        self.assertEqual(segments[1].pavement_type, "Beton")

    def test_short_segment_merging(self):
        """
        Skenario: Ada perubahan perkerasan di dekat batas blok (misal di STA 95),
        sehingga menyisakan segmen 95-100 m (panjang 5m < 10m).
        Segmen 5m tersebut harus digabung ke segmen tetangga dan diberi flag khusus.
        """
        frames = []
        for sta in range(0, 95, 5):
            frames.append(FrameRecord(frame_idx=len(frames), sta_m=float(sta), pavement_type="Hotmix", pavement_conf=0.9))
        frames.append(FrameRecord(frame_idx=len(frames), sta_m=95.0, pavement_type="Beton", pavement_conf=0.9))
        frames.append(FrameRecord(frame_idx=len(frames), sta_m=100.0, pavement_type="Beton", pavement_conf=0.9))

        segments = create_segments_from_frames(
            frames=frames,
            road_name="Ruas Uji",
            sta_start_m=0.0,
            base_segment_length_m=100.0,
            min_segment_length_m=10.0,
            smoothing_window=1,
            min_run_length=1
        )

        total_len = sum(s.length_m for s in segments)
        self.assertEqual(total_len, 100.0)
        self.assertTrue(any("Gabungan segmen pendek" in s.flag for s in segments))

    def test_manual_corrections(self):
        frames = [
            FrameRecord(frame_idx=0, sta_m=0.0, pavement_type="Hotmix", pavement_conf=0.9),
            FrameRecord(frame_idx=1, sta_m=100.0, pavement_type="Hotmix", pavement_conf=0.9),
        ]
        segments = create_segments_from_frames(frames, sta_start_m=0.0, base_segment_length_m=100.0)
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0].pavement_type, "Hotmix")

        overrides = [{
            "index": 0,
            "pavement_type": "Beton",
            "rut_depth_cm": 2.5
        }]
        corrected = apply_manual_corrections(segments, overrides)
        self.assertEqual(corrected[0].pavement_type, "Beton")
        self.assertEqual(corrected[0].rut_depth_cm, 2.5)
        self.assertEqual(corrected[0].flag, "Koreksi Manual")
        self.assertTrue(corrected[0].is_manual_override)

    def test_non_zero_sta_start_and_end(self):
        """
        Skenario: Video survei dimulai dari STA 542 m dan berakhir di STA 842 m (bukan dari 0 m).
        """
        frames = [
            FrameRecord(frame_idx=i, sta_m=float(542 + i * 20), pavement_type="Hotmix", pavement_conf=0.95)
            for i in range(16)  # 542 m s/d 842 m
        ]
        segments = create_segments_from_frames(
            frames=frames,
            road_name="Jl Anjungan",
            sta_start_m=None,
            base_segment_length_m=100.0,
            min_segment_length_m=10.0,
            smoothing_window=1,
            min_run_length=1
        )
        self.assertEqual(len(segments), 3)
        self.assertEqual(segments[0].sta_start, 542.0)
        self.assertEqual(segments[0].sta_end, 642.0)
        self.assertEqual(segments[-1].sta_end, 842.0)

    def test_sta_estimator_digit_refinement(self):
        """
        Menguji pemurnian lintasan STA dari pembacaan digit GPS (termasuk koreksi skala x10).
        """
        from models.sta_estimator import StaEstimator
        times = [0.0, 3.33, 6.67, 10.0, 13.33]
        raw_vals = [542.0, 575.0, 61.0, 642.0, 676.0]  # 61.0 adalah salah baca kehilangan 1 digit nol (seharusnya 610.0)
        cleaned_sta, anchor_t, anchor_v, speed = StaEstimator.refine_sta_readings(
            times=times,
            raw_vals=raw_vals,
            default_speed_mps=10.0
        )
        self.assertAlmostEqual(cleaned_sta[0], 542.0, delta=1.0)
        self.assertAlmostEqual(cleaned_sta[-1], 676.0, delta=1.0)
        self.assertAlmostEqual(cleaned_sta[2], 610.0, delta=5.0)


if __name__ == "__main__":
    unittest.main()
