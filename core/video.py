"""
Modul Pipeline Pemrosesan Video Survei Jalan dengan Kalibrasi Otomatis
dan Perekaman Video Hasil Inferensi (Web-Compatible H.264 MP4).
"""
import os
import cv2
import subprocess
import shutil
import tempfile
import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple
from .segmentation import FrameRecord, RoadSegment, create_segments_from_frames
from .sdi import calculate_sdi
from .recommendation import get_recommendation
from .calibration import auto_calibrate_camera, auto_calculate_frame_step

logger = logging.getLogger(__name__)


def calculate_iou(boxA: List[float], boxB: List[float]) -> float:
    """Menghitung Intersection over Union (IoU) antara dua bounding box [x1, y1, x2, y2]."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0.0, xB - xA) * max(0.0, yB - yA)
    boxAArea = max(0.0, boxA[2] - boxA[0]) * max(0.0, boxA[3] - boxA[1])
    boxBArea = max(0.0, boxB[2] - boxB[0]) * max(0.0, boxB[3] - boxB[1])
    unionArea = boxAArea + boxBArea - interArea

    return interArea / unionArea if unionArea > 0 else 0.0


class SimpleDamageTracker:
    """
    Tracker sederhana untuk menduplikasi objek kerusakan antar frame berurutan.
    Menghindari penghitungan ganda (double-counting) lubang & retak yang muncul di frame beruntun.
    """

    def __init__(self, iou_threshold: float = 0.25, max_age: int = 5):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.next_id = 1
        self.tracks: Dict[int, Dict[str, Any]] = {}

    def update(
        self,
        detections: List[Dict[str, Any]],
        frame_idx: int
    ) -> List[Dict[str, Any]]:
        for tid in list(self.tracks.keys()):
            self.tracks[tid]["age"] += 1
            if self.tracks[tid]["age"] > self.max_age:
                del self.tracks[tid]

        matched_track_ids = set()
        updated_detections = []

        for det in detections:
            bbox = det["bbox"]
            sdi_cat = det["sdi_category"]
            best_iou = 0.0
            best_tid = None

            for tid, trk in self.tracks.items():
                if tid in matched_track_ids:
                    continue
                if trk["sdi_category"] != sdi_cat:
                    continue

                iou = calculate_iou(bbox, trk["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_tid = tid

            if best_iou >= self.iou_threshold and best_tid is not None:
                matched_track_ids.add(best_tid)
                self.tracks[best_tid]["bbox"] = bbox
                self.tracks[best_tid]["age"] = 0
                self.tracks[best_tid]["last_seen_frame"] = frame_idx
                self.tracks[best_tid]["area_m2"] = max(self.tracks[best_tid]["area_m2"], det["area_m2"])
                det["track_id"] = best_tid
                det["is_new"] = False
            else:
                new_id = self.next_id
                self.next_id += 1
                self.tracks[new_id] = {
                    "bbox": bbox,
                    "class": det["class"],
                    "sdi_category": sdi_cat,
                    "age": 0,
                    "last_seen_frame": frame_idx,
                    "area_m2": det["area_m2"],
                }
                matched_track_ids.add(new_id)
                det["track_id"] = new_id
                det["is_new"] = True

            updated_detections.append(det)

        return updated_detections


def draw_annotations(
    frame: np.ndarray,
    detections: List[Dict[str, Any]],
    pavement_info: Dict[str, Any],
    sta_info: Dict[str, Any]
) -> np.ndarray:
    annotated = frame.copy()
    h, w = annotated.shape[:2]

    colors = {
        "crack": (0, 165, 255),      # Oranye BGR
        "pothole": (0, 0, 255),      # Merah BGR
        "other": (200, 200, 200),    # Abu-abu
    }

    for det in detections:
        bbox = det["bbox"]
        x1, y1, x2, y2 = [int(v) for v in bbox]
        cat = det.get("sdi_category", "other")
        color = colors.get(cat, (255, 255, 255))
        tid = det.get("track_id", "?")

        # Gambar Bounding Box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label_txt = f"ID:{tid} {det['class']} ({det['conf']:.2f})"
        (tw, th), _ = cv2.getTextSize(label_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated, (x1, max(0, y1 - th - 6)), (x1 + tw, y1), color, -1)
        cv2.putText(
            annotated, label_txt, (x1, max(th, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA
        )

    # HUD Banner bagian atas
    cv2.rectangle(annotated, (0, 0), (w, 48), (20, 20, 20), -1)
    sta_text = f"STA: {sta_info.get('sta_m', 0.0):.1f} m"
    pave_text = f"Perkerasan: {pavement_info.get('label', 'Unknown')} ({pavement_info.get('conf', 0.0)*100:.0f}%)"
    det_text = f"Deteksi: {len(detections)} kerusakan"
    full_hud = f"{sta_text}  |  {pave_text}  |  {det_text}"
    cv2.putText(annotated, full_hud, (15, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

    return annotated


def convert_to_web_h264(input_path: str, output_path: str) -> bool:
    """
    Mengonversi video ke format H.264 YUV420P menggunakan ffmpeg
    agar dapat diputar lancar langsung di HTML5 video player browser.
    """
    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        # Fallback jika ffmpeg tidak ditemukan di PATH
        shutil.copy(input_path, output_path)
        return False

    cmd = [
        ffmpeg_bin, "-y",
        "-i", input_path,
        "-vcodec", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "fast",
        "-crf", "24",
        output_path
    ]
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except Exception as e:
        logger.warning(f"Gagal convert ffmpeg h264: {e}. Menggunakan file asli.")
        shutil.copy(input_path, output_path)
        return False


def process_video_survey(
    video_path: str,
    damage_detector: Any,
    pavement_classifier: Any,
    sta_estimator: Any,
    road_name: str = "Ruas Survei",
    road_width_m: float = 6.0,
    base_segment_length_m: float = 100.0,
    crack_width_threshold_mm: float = 3.0,
    default_rut_depth_cm: float = 0.0,
    output_video_path: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    progress_callback: Optional[Any] = None
) -> Tuple[List[RoadSegment], List[Dict[str, Any]], Dict[str, Any], Optional[str]]:
    """
    Menjalankan alur lengkap pemrosesan video survei dengan KALIBRASI OTOMATIS
    dan pembuatan file video hasil inferensi beranotasi lengkap.

    Mengembalikan:
      (segments, preview_frames, stats, final_annotated_video_path)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Tidak dapat membuka file video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    h_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720

    auto_frame_step = auto_calculate_frame_step(fps)

    tracker = SimpleDamageTracker(iou_threshold=0.25, max_age=5)
    frame_records: List[FrameRecord] = []
    preview_frames: List[Dict[str, Any]] = []

    # Siapkan VideoWriter untuk merekam video hasil inferensi
    raw_video_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    
    # Hitung FPS output untuk video beranotasi (berjalan proporsional dengan sampling)
    out_fps = max(5.0, fps / float(auto_frame_step))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(raw_video_tmp, fourcc, out_fps, (w_orig, h_orig))

    processed_count = 0
    frame_idx = 0
    is_calibrated = False
    calib_info = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if not is_calibrated and frame is not None:
            calib_info = auto_calibrate_camera(frame.shape, road_width_m=road_width_m)
            if hasattr(damage_detector, "update_auto_calibration"):
                damage_detector.update_auto_calibration(
                    meter_per_pixel=calib_info["meter_per_pixel"],
                    crack_area_factors=calib_info["crack_area_factors"]
                )
            is_calibrated = True

        if frame_idx % auto_frame_step == 0:
            sta_res = sta_estimator.predict(frame=frame, frame_idx=frame_idx, fps=fps)
            sta_m = sta_res["sta_m"]

            pave_res = pavement_classifier.predict(frame)
            pavement_label = pave_res["label"]
            pavement_conf = pave_res["conf"]

            raw_dets = damage_detector.predict(frame)
            tracked_dets = tracker.update(raw_dets, frame_idx)

            pothole_tids = [d["track_id"] for d in tracked_dets if d["sdi_category"] == "pothole"]
            new_crack_m2 = sum(
                d["area_m2"] for d in tracked_dets 
                if d["sdi_category"] == "crack" and d.get("is_new", False)
            )
            crack_widths = [d["width_mm"] for d in tracked_dets if d["sdi_category"] == "crack"]
            avg_width = float(np.mean(crack_widths)) if crack_widths else 0.0

            rec = FrameRecord(
                frame_idx=frame_idx,
                sta_m=sta_m,
                pavement_type=pavement_label,
                pavement_conf=pavement_conf,
                crack_area_m2=new_crack_m2,
                crack_width_mm=avg_width,
                pothole_ids=pothole_tids,
                lat=sta_res.get("lat"),
                lon=sta_res.get("lon")
            )
            frame_records.append(rec)

            # Gambar anotasi untuk frame video inferensi
            annotated_frame = draw_annotations(frame, tracked_dets, pave_res, sta_res)
            video_writer.write(annotated_frame)

            # Sampel preview gambar untuk UI
            if len(preview_frames) < 10 or (len(raw_dets) > 0 and len(preview_frames) < 20):
                rgb_img = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                preview_frames.append({
                    "frame_idx": frame_idx,
                    "sta_m": sta_m,
                    "image": rgb_img,
                    "det_count": len(tracked_dets),
                    "pavement": pavement_label
                })

            processed_count += 1
            if progress_callback and total_frames > 0:
                prog = min(1.0, frame_idx / total_frames)
                progress_callback(prog, f"Menganalisis & merekam video inferensi (STA {sta_m:.1f} m)...")

        frame_idx += 1

    cap.release()
    video_writer.release()

    # Konversi video ke H.264 web compatible
    final_video_path = output_video_path or tempfile.NamedTemporaryFile(delete=False, suffix="_web.mp4").name
    convert_to_web_h264(raw_video_tmp, final_video_path)
    if os.path.exists(raw_video_tmp):
        try:
            os.remove(raw_video_tmp)
        except Exception:
            pass

    if not frame_records:
        return [], preview_frames, {"total_frames": total_frames, "processed_frames": 0}, None

    # Segmentasi Ruas
    segments = create_segments_from_frames(
        frames=frame_records,
        road_name=road_name,
        sta_start_m=0.0,
        base_segment_length_m=base_segment_length_m,
        min_segment_length_m=10.0,
        smoothing_window=5,
        min_run_length=3
    )

    # Perhitungan SDI per Segmen
    for seg in segments:
        seg_frames = [frame_records[i] for i in seg.frame_indices if i < len(frame_records)]

        total_crack_area_m2 = sum(f.crack_area_m2 for f in seg_frames)
        surface_area_m2 = max(1.0, seg.length_m * road_width_m)
        crack_area_pct = min(100.0, (total_crack_area_m2 / surface_area_m2) * 100.0)

        widths = [f.crack_width_mm for f in seg_frames if f.crack_width_mm > 0]
        avg_crack_width = float(np.mean(widths)) if widths else 0.0

        unique_potholes = len(set(pid for f in seg_frames for pid in f.pothole_ids))
        potholes_per_km = (unique_potholes * 1000.0) / max(1.0, seg.length_m)
        rut_cm = seg.rut_depth_cm if seg.rut_depth_cm > 0 else default_rut_depth_cm

        sdi_calc = calculate_sdi(
            crack_area_pct=crack_area_pct,
            avg_crack_width_mm=avg_crack_width,
            potholes_per_km=potholes_per_km,
            rut_depth_cm=rut_cm,
            crack_width_threshold_mm=crack_width_threshold_mm,
            config=config
        )

        rec_text, rec_note = get_recommendation(
            condition=sdi_calc.condition,
            pavement_type=seg.pavement_type,
            config=config
        )
        full_rec = f"{rec_text} ({rec_note})" if rec_note else rec_text

        seg.sdi_value = sdi_calc.final_sdi
        seg.condition = sdi_calc.condition
        seg.color = sdi_calc.color
        seg.crack_area_pct = round(crack_area_pct, 2)
        seg.avg_crack_width_mm = round(avg_crack_width, 2)
        seg.potholes_per_km = round(potholes_per_km, 2)
        seg.rut_depth_cm = round(rut_cm, 2)
        seg.unique_potholes = unique_potholes
        seg.total_crack_area_m2 = round(total_crack_area_m2, 3)
        seg.recommendation = full_rec

    total_len = sum(s.length_m for s in segments)
    stats = {
        "total_frames": total_frames,
        "processed_frames": processed_count,
        "fps": round(fps, 1),
        "total_segments": len(segments),
        "total_length_m": round(total_len, 1),
        "auto_calibration": calib_info.get("summary", "Kalibrasi Otomatis Aktif"),
    }

    return segments, preview_frames, stats, final_video_path
