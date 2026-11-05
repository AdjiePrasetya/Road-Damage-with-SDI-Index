"""
Modul Penyedia Data Demo dengan Video Hasil Inferensi Sintetis (Kecepatan Normal 1x).
"""
import os
import cv2
import tempfile
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from .segmentation import RoadSegment
from .sdi import calculate_sdi
from .recommendation import get_recommendation
from .video import convert_to_web_h264


def generate_synthetic_demo_video() -> str:
    """
    Membuat file video MP4 animasi sintetis dengan kecepatan normal 1x (30 FPS, durasi 6 detik)
    yang menampilkan pergerakan mobil, HUD STA, perkerasan, dan bounding box deteksi.
    """
    w, h = 640, 360
    fps = 30.0
    duration_sec = 6.0
    total_frames = int(fps * duration_sec) # 180 frame

    tmp_raw = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(tmp_raw, fourcc, fps, (w, h))

    for idx in range(total_frames):
        img = np.ones((h, w, 3), dtype=np.uint8) * 75
        sta_curr = (idx / total_frames) * 100.0

        # Marka jalan bergerak pada kecepatan normal
        offset = (idx * 6) % 60
        for y in range(-60 + offset, h, 60):
            cv2.line(img, (w // 2, y), (w // 2, y + 30), (255, 255, 255), 4)

        cv2.line(img, (80, 0), (40, h), (255, 255, 255), 3)
        cv2.line(img, (w - 80, 0), (w - 40, h), (255, 255, 255), 3)

        # Kerusakan jalan (retak, lubang, & alur dengan MASK KONTUR SEGMENTASI MURNI)
        cycle_progress = (idx % 90) / 90.0
        box_y = int(80 + cycle_progress * 180)
        box_scale = 1.0 + cycle_progress * 0.4
        overlay = img.copy()

        # 1. Kontur Segmentasi Retak (Polygon lekuk alami)
        bx1 = int(160 - cycle_progress * 20)
        bx2 = int(bx1 + 130 * box_scale)
        by1 = box_y
        by2 = int(by1 + 35 * box_scale)
        crack_pts = np.array([
            [int(bx1), int(by1 + 15 * box_scale)],
            [int(bx1 + 25 * box_scale), int(by1 + 2)],
            [int(bx1 + 60 * box_scale), int(by1 + 18 * box_scale)],
            [int(bx1 + 95 * box_scale), int(by1 + 4 * box_scale)],
            [int(bx2), int(by1 + 15 * box_scale)],
            [int(bx2 - 15 * box_scale), int(by2)],
            [int(bx1 + 80 * box_scale), int(by2 - 8 * box_scale)],
            [int(bx1 + 40 * box_scale), int(by2 + 5 * box_scale)],
            [int(bx1 + 12 * box_scale), int(by1 + 28 * box_scale)],
        ], dtype=np.int32)
        cv2.fillPoly(overlay, [crack_pts], (0, 165, 255))
        cv2.polylines(img, [crack_pts], isClosed=True, color=(0, 165, 255), thickness=2, lineType=cv2.LINE_AA)
        cv2.putText(img, "ID:1 crack_gt_3mm (0.89)", (bx1, max(16, by1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

        # 2. Kontur Segmentasi Lubang (Pothole berbentuk oval organik)
        center_x, center_y = int(390 + cycle_progress * 20), int(box_y + 20)
        rx, ry = int(32 * box_scale), int(20 * box_scale)
        pothole_pts = []
        for angle in range(0, 360, 25):
            rad = np.radians(angle)
            jitter = 1.0 + 0.12 * np.sin(angle * 3)
            pothole_pts.append([int(center_x + rx * jitter * np.cos(rad)), int(center_y + ry * jitter * np.sin(rad))])
        pothole_poly = np.array(pothole_pts, dtype=np.int32)
        cv2.fillPoly(overlay, [pothole_poly], (0, 0, 255))
        cv2.polylines(img, [pothole_poly], isClosed=True, color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA)
        cv2.putText(img, "ID:2 pothole (0.94)", (center_x - rx, max(16, center_y - ry - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

        # Blend semi-transparent mask segmentasi
        cv2.addWeighted(overlay, 0.40, img, 0.60, 0, img)

        # HUD Top
        cv2.rectangle(img, (0, 0), (w, 38), (20, 20, 20), -1)
        pave = "Hotmix" if sta_curr < 60 else "Beton"
        hud_txt = f"STA: {sta_curr:.1f} m  |  Perkerasan: {pave}  |  Deteksi: 2 kerusakan"
        cv2.putText(img, hud_txt, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)

        writer.write(img)

    writer.release()

    final_web = tempfile.NamedTemporaryFile(delete=False, suffix="_web.mp4").name
    convert_to_web_h264(tmp_raw, final_web, target_fps=fps)
    if os.path.exists(tmp_raw):
        try:
            os.remove(tmp_raw)
        except Exception:
            pass

    return final_web


def generate_demo_survey_data(
    road_name: str = "Jalan Ilyas Bedul (Demo)",
    sta_start: float = 0.0,
    road_width_m: float = 6.0,
    config: Dict[str, Any] = None
) -> Tuple[List[RoadSegment], List[Dict[str, Any]], Dict[str, Any], str]:
    raw_specs = [
        {
            "sta_start": 0.0,
            "sta_end": 60.0,
            "pavement": "Hotmix",
            "crack_pct": 12.0,
            "crack_w": 3.5,
            "potholes_km": 15.0,
            "rut_cm": 1.5,
            "unique_potholes": 1,
            "flag": "Dipotong (Perubahan Perkerasan)",
        },
        {
            "sta_start": 60.0,
            "sta_end": 100.0,
            "pavement": "Beton",
            "crack_pct": 8.0,
            "crack_w": 2.0,
            "potholes_km": 12.0,
            "rut_cm": 1.5,
            "unique_potholes": 1,
            "flag": "Dipotong (Perubahan Perkerasan)",
        },
        {
            "sta_start": 100.0,
            "sta_end": 200.0,
            "pavement": "Hotmix",
            "crack_pct": 4.5,
            "crack_w": 1.5,
            "potholes_km": 5.0,
            "rut_cm": 0.5,
            "unique_potholes": 1,
            "flag": "Normal",
        },
        {
            "sta_start": 200.0,
            "sta_end": 300.0,
            "pavement": "Hotmix",
            "crack_pct": 35.0,
            "crack_w": 4.0,
            "potholes_km": 15.0,
            "rut_cm": 3.5,
            "unique_potholes": 2,
            "flag": "Normal",
        },
        {
            "sta_start": 300.0,
            "sta_end": 400.0,
            "pavement": "Beton",
            "crack_pct": 5.0,
            "crack_w": 2.0,
            "potholes_km": 0.0,
            "rut_cm": 0.0,
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

    preview_frames = []
    for i, seg in enumerate(segments[:4]):
        img = np.ones((480, 640, 3), dtype=np.uint8) * 80
        overlay = img.copy()
        cv2.line(img, (320, 0), (320, 480), (255, 255, 255), 4)

        if seg.sdi_value > 50:
            crack_pts = np.array([
                [200, 240], [250, 220], [310, 245], [370, 225],
                [385, 290], [330, 280], [270, 320], [210, 300]
            ], dtype=np.int32)
            cv2.fillPoly(overlay, [crack_pts], (0, 165, 255))
            cv2.polylines(img, [crack_pts], isClosed=True, color=(0, 165, 255), thickness=2, lineType=cv2.LINE_AA)
            cv2.putText(img, "crack_gt_3mm (0.88)", (200, 212), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

        if seg.unique_potholes > 0:
            pothole_pts = np.array([
                [270, 350], [305, 335], [335, 350], [345, 385],
                [320, 405], [285, 400], [260, 375]
            ], dtype=np.int32)
            cv2.fillPoly(overlay, [pothole_pts], (0, 0, 255))
            cv2.polylines(img, [pothole_pts], isClosed=True, color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA)
            cv2.putText(img, "pothole (0.92)", (260, 325), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

        if seg.rut_depth_cm > 0:
            rut_pts = np.array([
                [150, 200], [185, 200], [195, 370], [155, 370]
            ], dtype=np.int32)
            cv2.fillPoly(overlay, [rut_pts], (255, 0, 200))
            cv2.polylines(img, [rut_pts], isClosed=True, color=(255, 0, 200), thickness=2, lineType=cv2.LINE_AA)
            cv2.putText(img, "rut_1_3cm (0.86)", (150, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1, cv2.LINE_AA)

        cv2.addWeighted(overlay, 0.40, img, 0.60, 0, img)

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

    demo_video_path = generate_synthetic_demo_video()

    stats = {
        "total_frames": 1200,
        "processed_frames": 120,
        "fps": 30.0,
        "total_segments": len(segments),
        "total_length_m": 400.0,
        "auto_calibration": "Resolusi 1280x720 px -> 0.00625 m/px (Otomatis)",
    }

    return segments, preview_frames, stats, demo_video_path
