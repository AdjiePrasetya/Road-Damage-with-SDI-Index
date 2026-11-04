"""
Modul Penyedia Data Demo untuk Pengujian Dashboard Tanpa Video Asli.
Membangkitkan data segmen realistis dengan contoh kasus perubahan perkerasan,
variasi kondisi SDI, dan data frame tiruan.
"""
import numpy as np
from typing import List, Dict, Any, Tuple
from .segmentation import RoadSegment
from .sdi import calculate_sdi
from .recommendation import get_recommendation


def generate_demo_survey_data(
    road_name: str = "Jalan Ilyas Bedul (Demo)",
    sta_start: float = 0.0,
    road_width_m: float = 6.0,
    config: Dict[str, Any] = None
) -> Tuple[List[RoadSegment], List[Dict[str, Any]], Dict[str, Any]]:
    """
    Menghasilkan data survei contoh untuk pengujian langsung di Streamlit.
    Menyertakan kasus:
      - Segmen 0 - 60 m   : Hotmix (SDI 125, Rusak Ringan)
      - Segmen 60 - 100 m : Beton (SDI 95, Sedang)
      - Segmen 100 - 200 m: Hotmix (SDI 25, Baik)
      - Segmen 200 - 300 m: Hotmix (SDI 175, Rusak Berat)
      - Segmen 300 - 400 m: Beton (SDI 45, Baik)
    """
    raw_specs = [
        {
            "sta_start": 0.0,
            "sta_end": 60.0,
            "pavement": "Hotmix",
            "crack_pct": 12.0,      # SDI1 = 20
            "crack_w": 3.5,         # SDI2 = 40 (lebar > 3mm)
            "potholes_km": 15.0,    # SDI3 = 40 + 75 = 115
            "rut_cm": 1.5,          # SDI4 = 115 + 10 = 125 (Rusak Ringan)
            "unique_potholes": 1,
            "flag": "Dipotong (Perubahan Perkerasan)",
        },
        {
            "sta_start": 60.0,
            "sta_end": 100.0,
            "pavement": "Beton",
            "crack_pct": 8.0,       # SDI1 = 5
            "crack_w": 2.0,         # SDI2 = 5
            "potholes_km": 12.0,    # SDI3 = 5 + 75 = 80
            "rut_cm": 1.5,          # SDI4 = 80 + 10 = 90 -> Sedang
            "unique_potholes": 1,
            "flag": "Dipotong (Perubahan Perkerasan)",
        },
        {
            "sta_start": 100.0,
            "sta_end": 200.0,
            "pavement": "Hotmix",
            "crack_pct": 4.5,       # SDI1 = 5
            "crack_w": 1.5,         # SDI2 = 5
            "potholes_km": 5.0,     # SDI3 = 5 + 15 = 20
            "rut_cm": 0.5,          # SDI4 = 20 + 2.5 = 22.5 (Baik)
            "unique_potholes": 1,
            "flag": "Normal",
        },
        {
            "sta_start": 200.0,
            "sta_end": 300.0,
            "pavement": "Hotmix",
            "crack_pct": 35.0,      # SDI1 = 40
            "crack_w": 4.0,         # SDI2 = 80
            "potholes_km": 15.0,    # SDI3 = 80 + 75 = 155
            "rut_cm": 3.5,          # SDI4 = 155 + 20 = 175 (Rusak Berat)
            "unique_potholes": 2,
            "flag": "Normal",
        },
        {
            "sta_start": 300.0,
            "sta_end": 400.0,
            "pavement": "Beton",
            "crack_pct": 5.0,       # SDI1 = 5
            "crack_w": 2.0,         # SDI2 = 5
            "potholes_km": 0.0,     # SDI3 = 5
            "rut_cm": 0.0,          # SDI4 = 5 (Baik)
            "unique_potholes": 0,
            "flag": "Normal",
        },
    ]

    segments: List[RoadSegment] = []

    for sp in raw_specs:
        length_m = sp["sta_end"] - sp["sta_start"]
        sdi_res = calculate_sdi(
            crack_area_pct=sp["crack_pct"],
            avg_crack_width_mm=sp["crack_w"],
            potholes_per_km=sp["potholes_km"],
            rut_depth_cm=sp["rut_cm"],
            crack_width_threshold_mm=3.0,
            config=config
        )

        rec_text, rec_note = get_recommendation(
            condition=sdi_res.condition,
            pavement_type=sp["pavement"],
            config=config
        )
        full_rec = f"{rec_text} ({rec_note})" if rec_note else rec_text

        seg = RoadSegment(
            road_name=road_name,
            sta_start=sp["sta_start"],
            sta_end=sp["sta_end"],
            length_m=length_m,
            pavement_type=sp["pavement"],
            sdi_value=sdi_res.final_sdi,
            condition=sdi_res.condition,
            recommendation=full_rec,
            crack_area_pct=sp["crack_pct"],
            avg_crack_width_mm=sp["crack_w"],
            potholes_per_km=sp["potholes_km"],
            rut_depth_cm=sp["rut_cm"],
            flag=sp["flag"],
            color=sdi_res.color,
            unique_potholes=sp["unique_potholes"],
            total_crack_area_m2=round((sp["crack_pct"] / 100.0) * (length_m * road_width_m), 2)
        )
        segments.append(seg)

    # Buat preview frame sintetis (kanvas gambar sintetis dengan anotasi)
    preview_frames = []
    for i, seg in enumerate(segments[:4]):
        img = np.ones((480, 640, 3), dtype=np.uint8) * 80  # Background abu-abu jalan
        # Gambar garis marka jalan
        cv2.line(img, (320, 0), (320, 480), (255, 255, 255), 4)
        cv2.line(img, (100, 0), (50, 480), (255, 255, 255), 3)
        cv2.line(img, (540, 0), (590, 480), (255, 255, 255), 3)

        # Gambar kerusakan dummy jika ada
        if seg.sdi_value > 50:
            # Box retak (Oranye)
            cv2.rectangle(img, (200, 220), (380, 320), (255, 165, 0), 2)
            cv2.putText(img, "D20 Alligator Crack (0.88)", (200, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 1)

        if seg.unique_potholes > 0:
            # Box lubang (Merah)
            cv2.rectangle(img, (260, 340), (340, 400), (255, 0, 0), 2)
            cv2.putText(img, "D40 Pothole (0.92)", (260, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # HUD atas
        cv2.rectangle(img, (0, 0), (640, 40), (20, 20, 20), -1)
        hud_txt = f"STA: {seg.sta_start:.0f}m - {seg.sta_end:.0f}m | {seg.pavement_type} | SDI: {seg.sdi_value:.1f} ({seg.condition})"
        cv2.putText(img, hud_txt, (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

        preview_frames.append({
            "frame_idx": i * 100,
            "sta_m": seg.sta_start,
            "image": img,
            "det_count": 2 if seg.sdi_value > 50 else 0,
            "pavement": seg.pavement_type
        })

    stats = {
        "total_frames": 1200,
        "processed_frames": 120,
        "fps": 30.0,
        "total_segments": len(segments),
        "total_length_m": 400.0,
    }

    return segments, preview_frames, stats

