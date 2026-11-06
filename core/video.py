"""
Modul Pipeline Pemrosesan Video Survei Jalan dengan Kalibrasi Otomatis
dan Perekaman Video Hasil Inferensi Real-Time (1:1 Durasi & Kecepatan Normal, H.264 MP4).
Terintegrasi dengan 3 Model Segmentasi Terbaru (Cracks, Potholes, Rutting Otomatis).
"""
import os
import cv2
import torch
import subprocess
import shutil
import tempfile
import numpy as np
import logging
import threading
import queue
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
    Tracker untuk menduplikasi objek kerusakan antar frame berurutan.
    Menghindari penghitungan ganda (double-counting) retak, lubang, & alur yang muncul beruntun.
    """

    def __init__(self, iou_threshold: float = 0.25, max_age: int = 8):
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
                if det.get("polygon") is not None:
                    self.tracks[best_tid]["polygon"] = det["polygon"]
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
                    "polygon": det.get("polygon"),
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

    # Palet warna kategori SDI (BGR)
    colors = {
        "crack": (0, 165, 255),      # Oranye (Retak)
        "pothole": (0, 0, 255),      # Merah (Lubang)
        "rutting": (255, 0, 200),    # Magenta (Alur/Rutting)
        "other": (200, 200, 200),
    }

    # Overlay mask semi-transparan untuk hasil segmentasi murni
    overlay = annotated.copy()
    has_any_poly = False

    for det in detections:
        cat = det.get("sdi_category", "other")
        color = colors.get(cat, (0, 255, 0))
        tid = det.get("track_id", "?")
        label_txt = f"ID:{tid} {det['class']} ({det['conf']:.2f})"

        polygon = det.get("polygon")
        if polygon is not None and len(polygon) >= 3:
            pts = np.int32(polygon).reshape((-1, 1, 2))
            # 1. Fill mask semi-transparan (Segmentasi)
            cv2.fillPoly(overlay, [pts], color)
            # 2. Garis kontur tepi segmentasi mengikuti lekuk riil kerusakan jalan (Tanpa Kotak Bounding Box)
            cv2.polylines(annotated, [pts], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)
            has_any_poly = True

            # Titik posisi label tag di bagian atas kontur
            min_y = int(np.min(pts[:, 0, 1]))
            min_x = int(np.min(pts[:, 0, 0]))
            label_pos_x = max(5, min(w - 180, min_x))
            label_pos_y = max(22, min_y - 6)

            (tw, th), _ = cv2.getTextSize(label_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(annotated, (label_pos_x, label_pos_y - th - 4), (label_pos_x + tw + 6, label_pos_y + 2), color, -1)
            cv2.putText(annotated, label_txt, (label_pos_x + 3, label_pos_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
        else:
            # Fallback jika polygon belum tersedia: gunakan bentuk oval/elips halus (bukan kotak kaku)
            bbox = det.get("bbox")
            if bbox:
                x1, y1, x2, y2 = [int(v) for v in bbox]
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                rx, ry = max(6, (x2 - x1) // 2), max(6, (y2 - y1) // 2)
                cv2.ellipse(overlay, (cx, cy), (rx, ry), 0, 0, 360, color, -1)
                cv2.ellipse(annotated, (cx, cy), (rx, ry), 0, 0, 360, color, 2, cv2.LINE_AA)
                has_any_poly = True

                (tw, th), _ = cv2.getTextSize(label_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.rectangle(annotated, (x1, max(0, y1 - th - 6)), (x1 + tw + 6, y1), color, -1)
                cv2.putText(annotated, label_txt, (x1 + 3, max(th, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

    if has_any_poly:
        cv2.addWeighted(overlay, 0.40, annotated, 0.60, 0, annotated)

    # HUD Banner bagian atas
    cv2.rectangle(annotated, (0, 0), (w, 48), (20, 20, 20), -1)
    sta_text = f"STA: {sta_info.get('sta_m', 0.0):.1f} m"
    pave_text = f"Perkerasan: {pavement_info.get('label', 'Unknown')}"
    det_text = f"Kerusakan: {len(detections)} segmen"
    full_hud = f"{sta_text}  |  {pave_text}  |  {det_text}"
    cv2.putText(annotated, full_hud, (15, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)

    return annotated


class AsyncVideoWriter:
    """
    Perekam video asinkron multi-threading berbasis antrean (Queue).
    Memisahkan operasi I/O penulisan frame ke disk dari loop inferensi GPU
    sehingga GPU tidak pernah terhambat oleh kecepatan penulisan disk.
    """

    def __init__(
        self,
        filename: str,
        fourcc: int,
        fps: float,
        frame_size: Tuple[int, int],
        queue_size: int = 128
    ):
        self.filename = filename
        self.writer = cv2.VideoWriter(filename, fourcc, fps, frame_size)
        self.queue = queue.Queue(maxsize=queue_size)
        self.stopped = False
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _worker(self):
        while True:
            frame = self.queue.get()
            if frame is None:
                self.queue.task_done()
                break
            try:
                self.writer.write(frame)
            except Exception as e:
                logger.error(f"Error menulis frame video: {e}")
            self.queue.task_done()

    def write(self, frame: np.ndarray) -> None:
        if not self.stopped:
            try:
                self.queue.put(frame, block=True, timeout=5.0)
            except queue.Full:
                logger.warning("Antrean video writer penuh, frame dilewati.")

    def release(self) -> None:
        self.stopped = True
        try:
            self.queue.put(None, timeout=2.0)
            self.thread.join(timeout=10.0)
        except Exception:
            pass
        self.writer.release()


def convert_to_web_h264(input_path: str, output_path: str, target_fps: float = 30.0) -> bool:
    """Mengonversi video ke format H.264 web-ready dengan akselerasi GPU NVENC jika tersedia."""
    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        shutil.copy(input_path, output_path)
        return False

    has_nvenc = False
    try:
        chk = subprocess.run([ffmpeg_bin, "-encoders"], capture_output=True, text=True, timeout=5)
        if "h264_nvenc" in chk.stdout:
            has_nvenc = True
    except Exception:
        pass

    if has_nvenc and torch.cuda.is_available():
        # GPU NVENC encoding super cepat (200+ FPS) dengan kompresi optimal
        cmd = [
            ffmpeg_bin, "-y",
            "-i", input_path,
            "-r", str(round(target_fps, 2)),
            "-c:v", "h264_nvenc",
            "-preset", "p4",
            "-cq", "26",
            "-pix_fmt", "yuv420p",
            output_path
        ]
    else:
        # CPU libx264 encoding dengan multi-threading maksimal
        cmd = [
            ffmpeg_bin, "-y",
            "-i", input_path,
            "-r", str(round(target_fps, 2)),
            "-c:v", "libx264",
            "-preset", "faster",
            "-crf", "26",
            "-pix_fmt", "yuv420p",
            "-threads", str(os.cpu_count() or 8),
            output_path
        ]

    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except Exception as e:
        logger.warning(f"Gagal convert ffmpeg: {e}. Menggunakan file mentah.")
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
    output_video_path: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    progress_callback: Optional[Any] = None
) -> Tuple[List[RoadSegment], List[Dict[str, Any]], Dict[str, Any], Optional[str]]:
    """
    Menjalankan alur pemrosesan video survei:
    - 3 Model Segmentasi Aktif: Cracks, Potholes, Rutting (Otomatis).
    - Akselerasi GPU NVIDIA (CUDA) diprioritaskan penuh, atau utilisasi CPU maksimal jika tanpa GPU.
    - Resolusi dan encoding video dioptimalkan agar ringan dan bebas MemoryError.
    - Video hasil disimpan permanen di folder lokal 'output_survei/'.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Tidak dapat membuka file video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    h_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720

    # Optimasi dimensi output: Batasi maks 1280px agar hemat memori RAM dan ukuran file ringan
    out_w, out_h = w_orig, h_orig
    max_dim = 1280
    if max(w_orig, h_orig) > max_dim:
        scale = max_dim / float(max(w_orig, h_orig))
        out_w = int(w_orig * scale)
        out_h = int(h_orig * scale)
        out_w = out_w if out_w % 2 == 0 else out_w - 1
        out_h = out_h if out_h % 2 == 0 else out_h - 1

    infer_step = max(3, min(int(round(fps * 0.33)), 15))

    tracker = SimpleDamageTracker(iou_threshold=0.25, max_age=infer_step * 2)
    frame_records: List[FrameRecord] = []
    preview_frames: List[Dict[str, Any]] = []

    # Map untuk melacak estimasi rutting kedalaman per frame
    frame_rut_depths: Dict[int, float] = {}

    raw_video_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name

    # VideoWriter asinkron menggunakan resolusi yang dioptimalkan
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = AsyncVideoWriter(raw_video_tmp, fourcc, fps, (out_w, out_h))

    # Pemanasan model GPU (Warm-up) agar CUDA kernel siap sebelum loop inferensi
    if torch.cuda.is_available():
        try:
            from core.device import warmup_gpu_models
            warmup_gpu_models([damage_detector, pavement_classifier, sta_estimator])
        except Exception:
            pass

    frame_idx = 0
    is_calibrated = False
    calib_info = {}

    active_detections: List[Dict[str, Any]] = []
    active_pavement: Dict[str, Any] = {"label": "Hotmix", "conf": 1.0}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Optimasi Kinerja GPU/CPU: Resize langsung di awal jika dimensi asli > 1280px
        # Menghilangkan beban redundansi penggambaran dan komputasi di CPU
        if (w_orig, h_orig) != (out_w, out_h):
            frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)

        if not is_calibrated and frame is not None:
            calib_info = auto_calibrate_camera(frame.shape, road_width_m=road_width_m)
            if hasattr(damage_detector, "update_auto_calibration"):
                damage_detector.update_auto_calibration(
                    meter_per_pixel=calib_info["meter_per_pixel"],
                    crack_area_factors=calib_info.get("crack_area_factors")
                )
            is_calibrated = True

        # Akselerasi STA: Hanya inferensi model visual pada keyframe, posisi kumulatif dihitung instan setiap frame
        frame_for_sta = frame if (frame_idx % infer_step == 0) else None
        sta_res = sta_estimator.predict(frame=frame_for_sta, frame_idx=frame_idx, fps=fps)
        sta_m = sta_res["sta_m"]

        if frame_idx % infer_step == 0:
            active_pavement = pavement_classifier.predict(frame)
            raw_dets = damage_detector.predict(frame)
            active_detections = tracker.update(raw_dets, frame_idx)

            # Lubang (potholes)
            pothole_tids = [d["track_id"] for d in active_detections if d["sdi_category"] == "pothole"]

            # Retak baru (cracks area & lebar)
            new_crack_m2 = sum(
                d["area_m2"] for d in active_detections 
                if d["sdi_category"] == "crack" and d.get("is_new", False)
            )
            crack_widths = [d["width_mm"] for d in active_detections if d["sdi_category"] == "crack" and d.get("width_mm", 0) > 0]
            avg_width = float(np.mean(crack_widths)) if crack_widths else 0.0

            # Alur (rutting) OTOMATIS DARI MODEL
            rut_depths = [d.get("rut_depth_cm", 0.0) for d in active_detections if d["sdi_category"] == "rutting"]
            max_frame_rut = max(rut_depths) if rut_depths else 0.0
            frame_rut_depths[frame_idx] = max_frame_rut

            rec = FrameRecord(
                frame_idx=frame_idx,
                sta_m=sta_m,
                pavement_type=active_pavement["label"],
                pavement_conf=active_pavement["conf"],
                crack_area_m2=new_crack_m2,
                crack_width_mm=avg_width,
                pothole_ids=pothole_tids,
                lat=sta_res.get("lat"),
                lon=sta_res.get("lon")
            )
            frame_records.append(rec)

        annotated_frame = draw_annotations(frame, active_detections, active_pavement, sta_res)
        video_writer.write(annotated_frame)

        if frame_idx % infer_step == 0 and (len(preview_frames) < 10 or (len(active_detections) > 0 and len(preview_frames) < 20)):
            preview_frames.append({
                "frame_idx": frame_idx,
                "sta_m": sta_m,
                "image": cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB),
                "det_count": len(active_detections),
                "pavement": active_pavement["label"]
            })

        if progress_callback and total_frames > 0 and frame_idx % 15 == 0:
            prog = min(1.0, frame_idx / total_frames)
            progress_callback(prog, f"Memproses video durasi normal (Frame {frame_idx}/{total_frames} • STA {sta_m:.1f} m)...")

        frame_idx += 1

    cap.release()
    video_writer.release()

    # Simpan hasil video permanen di folder lokal project 'output_survei/'
    output_dir = os.path.abspath("output_survei")
    os.makedirs(output_dir, exist_ok=True)
    clean_road = "".join(c for c in road_name if c.isalnum() or c in (' ', '_', '-')).strip().replace(' ', '_')
    default_out_path = os.path.join(output_dir, f"Hasil_Survei_{clean_road}.mp4")
    final_video_path = output_video_path or default_out_path

    convert_to_web_h264(raw_video_tmp, final_video_path, target_fps=fps)
    if os.path.exists(raw_video_tmp):
        try:
            os.remove(raw_video_tmp)
        except Exception:
            pass

    if not frame_records:
        return [], preview_frames, {"total_frames": total_frames, "processed_frames": 0}, None

    # Segmentasi Ruas PKRMS
    segments = create_segments_from_frames(
        frames=frame_records,
        road_name=road_name,
        sta_start_m=0.0,
        base_segment_length_m=base_segment_length_m,
        min_segment_length_m=10.0,
        smoothing_window=5,
        min_run_length=3
    )

    # Perhitungan SDI per Segmen Sesuai Panduan PKRMS Tabel 9 & 10
    for seg in segments:
        seg_frames = [frame_records[i] for i in seg.frame_indices if i < len(frame_records)]

        # Level 1: % Luas Retak (dari mask polygon riil)
        total_crack_area_m2 = sum(f.crack_area_m2 for f in seg_frames)
        surface_area_m2 = max(1.0, seg.length_m * road_width_m)
        crack_area_pct = min(100.0, (total_crack_area_m2 / surface_area_m2) * 100.0)

        # Level 2: Lebar Retak (dari kelas model crack_lt_1mm, crack_1_3mm, crack_gt_3mm)
        widths = [f.crack_width_mm for f in seg_frames if f.crack_width_mm > 0]
        avg_crack_width = float(np.mean(widths)) if widths else 0.0

        # Level 3: Lubang unik per km
        unique_potholes = len(set(pid for f in seg_frames for pid in f.pothole_ids))
        potholes_per_km = (unique_potholes * 1000.0) / max(1.0, seg.length_m)

        # Level 4: KEDALAMAN ALUR OTOMATIS DARI MODEL RUTTING
        seg_rut_depths = [frame_rut_depths.get(f.frame_idx, 0.0) for f in seg_frames if frame_rut_depths.get(f.frame_idx, 0.0) > 0]
        if seg_rut_depths:
            # Gunakan nilai representatif alur terdeteksi
            auto_rut_depth_cm = float(np.percentile(seg_rut_depths, 75))
        else:
            auto_rut_depth_cm = 0.0

        sdi_calc = calculate_sdi(
            crack_area_pct=crack_area_pct,
            avg_crack_width_mm=avg_crack_width,
            potholes_per_km=potholes_per_km,
            rut_depth_cm=auto_rut_depth_cm,
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
        seg.rut_depth_cm = round(auto_rut_depth_cm, 2)
        seg.unique_potholes = unique_potholes
        seg.total_crack_area_m2 = round(total_crack_area_m2, 3)
        seg.recommendation = full_rec

    total_len = sum(s.length_m for s in segments)
    stats = {
        "total_frames": total_frames,
        "processed_frames": len(frame_records),
        "fps": round(fps, 1),
        "total_segments": len(segments),
        "total_length_m": round(total_len, 1),
        "auto_calibration": calib_info.get("summary", "Kalibrasi Otomatis Aktif"),
    }

    return segments, preview_frames, stats, final_video_path
