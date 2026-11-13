"""
Adapter Estimasi STA dan Posisi GPS (StaEstimator) Berbasis Model gps_sta_model/weights/best.pt.
Menggunakan model deteksi digit YOLO pada layar Garmin GPS (dengan ROI dari roi.json)
untuk membaca nilai STA awal jalan hingga akhir jalan secara akurat dan otomatis.
Antarmuka seragam: StaEstimator.predict(frame, frame_idx, fps) -> {sta_m, lat, lon, ...}
"""
import os
import json
import logging
from typing import Dict, Any, Optional, Tuple, List
import cv2
import numpy as np
from core.device import get_optimal_device

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

logger = logging.getLogger(__name__)


class StaEstimator:
    """
    Adapter model gps_sta_model/weights/best.pt untuk membaca digit STA langsung dari
    perangkat GPS pada video survei jalan menggunakan Region of Interest (roi.json)
    serta pemfilteran trajektori skala-adaptif (RANSAC & Monotonic Filtering).
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        weights_path: Optional[str] = None,
        roi_path: Optional[str] = None,
        sta_start_m: float = 0.0,
        default_speed_kmh: float = 30.0,
        gps_start_coord: Optional[Tuple[float, float]] = None,
        device: str = "auto",
        **kwargs
    ):
        self.sta_start_m = sta_start_m
        self.default_speed_kmh = default_speed_kmh
        self.gps_start_coord = gps_start_coord or (-6.2088, 106.8456)
        self.device, self.device_info = get_optimal_device(device)

        target_path = model_path or weights_path
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # Prioritaskan model terbaru gps_sta_model/weights/best.pt
        candidates = [
            target_path,
            os.path.join(base_dir, target_path) if target_path else None,
            os.path.join(base_dir, "gps_sta_model", "weights", "best.pt"),
            os.path.join("gps_sta_model", "weights", "best.pt"),
            os.path.join(base_dir, "gps_sta_model", "runs", "detect", "train", "weights", "best.pt"),
            os.path.join("gps_sta_model", "runs", "detect", "train", "weights", "best.pt"),
            os.path.join(base_dir, "gps_sta_model", "weights", "last.pt"),
            os.path.join("gps_sta_model", "weights", "last.pt"),
            os.path.join(base_dir, "gps_sta_model", "weights", "yolo26n.pt"),
            os.path.join("gps_sta_model", "weights", "yolo26n.pt"),
        ]

        self.model = None
        self.model_loaded_path = None
        self.is_digit_model = False

        if YOLO is not None:
            for p in candidates:
                if p and os.path.exists(p):
                    try:
                        self.model = YOLO(p)
                        self.model_loaded_path = p
                        names_map = getattr(self.model, "names", {}) or {}
                        names_set = set(str(v) for v in names_map.values())
                        if "0" in names_set and "1" in names_set:
                            self.is_digit_model = True
                        logger.info(
                            f"Berhasil memuat model StaEstimator dari: {p} "
                            f"(Digit STA Model: {self.is_digit_model}, Device: {self.device})"
                        )
                        break
                    except Exception as e:
                        logger.warning(f"Gagal memuat model GPS/STA dari {p}: {e}")

        # Muat konfigurasi ROI (Region of Interest layar GPS) dari roi.json
        self.roi_norm: List[float] = [0.6865711666666667, 0.0, 0.9811643333333334, 0.34869447058823533]
        roi_candidates = [
            roi_path,
            os.path.join(base_dir, roi_path) if roi_path else None,
            os.path.join(base_dir, "gps_sta_model", "weights", "roi.json"),
            os.path.join("gps_sta_model", "weights", "roi.json"),
            os.path.join(base_dir, "gps_sta_model", "roi.json"),
            os.path.join("gps_sta_model", "roi.json"),
        ]
        for rp in roi_candidates:
            if rp and os.path.exists(rp):
                try:
                    with open(rp, "r", encoding="utf-8") as f:
                        roi_data = json.load(f)
                        if "roi_norm" in roi_data and len(roi_data["roi_norm"]) == 4:
                            self.roi_norm = [float(x) for x in roi_data["roi_norm"]]
                            logger.info(f"Berhasil memuat ROI STA dari {rp}: {self.roi_norm}")
                            break
                except Exception as e:
                    logger.warning(f"Gagal memuat roi.json dari {rp}: {e}")

        self.reset()

    def reset(self) -> None:
        """Mereset state pelacakan STA sebelum memulai pemrosesan video baru."""
        self._last_known_sta: Optional[float] = None
        self._last_known_time_s: float = 0.0
        self._detected_start_sta: Optional[float] = None
        self._last_known_lat: float = self.gps_start_coord[0]
        self._last_known_lon: float = self.gps_start_coord[1]
        self._last_features_count: int = 0
        self._last_raw_sta_str: str = ""
        self._est_speed_mps: float = (self.default_speed_kmh * 1000.0) / 3600.0
        self._raw_history: List[Tuple[int, float, Optional[float]]] = []
        self._anchor_times: Optional[np.ndarray] = None
        self._anchor_stas: Optional[np.ndarray] = None
        self._global_speed_mps: Optional[float] = None

    def _crop_roi(self, frame: np.ndarray) -> np.ndarray:
        """Memotong area ROI layar Garmin GPS dari frame penuh."""
        h, w = frame.shape[:2]
        # Jika frame sudah sangat kecil (misal hasil crop atau dummy frame uji), gunakan langsung
        if w < 300 or h < 200:
            return frame
        x1 = max(0, min(w - 1, int(round(self.roi_norm[0] * w))))
        y1 = max(0, min(h - 1, int(round(self.roi_norm[1] * h))))
        x2 = max(x1 + 1, min(w, int(round(self.roi_norm[2] * w))))
        y2 = max(y1 + 1, min(h, int(round(self.roi_norm[3] * h))))
        return frame[y1:y2, x1:x2]

    def _read_raw_sta_from_frame(self, frame: np.ndarray) -> Tuple[Optional[float], int, str]:
        """
        Menjalankan model deteksi digit STA (best.pt) pada ROI frame,
        melakukan deduplikasi slot horizontal (1D IoMin), mengurutkan digit dari kiri ke kanan,
        dan mengonversi menjadi nilai float STA (meter).
        """
        if self.model is None or frame is None:
            return None, 0, ""

        try:
            roi_img = self._crop_roi(frame)
            results = self.model.predict(
                source=roi_img,
                conf=0.18,
                iou=0.45,
                imgsz=640 if self.is_digit_model else 320,
                agnostic_nms=True,
                device=self.device,
                verbose=False
            )
            if not results or len(results) == 0:
                return None, 0, ""

            r = results[0]
            boxes = getattr(r, "boxes", None)
            if boxes is None or len(boxes) == 0:
                return None, 0, ""

            if not self.is_digit_model:
                return None, len(boxes), ""

            names_map = r.names or self.model.names
            raw_boxes = []
            for b in boxes:
                cls_id = int(b.cls[0].item())
                name = str(names_map.get(cls_id, ""))
                conf = float(b.conf[0].item())
                bx1, by1, bx2, by2 = [float(v) for v in b.xyxy[0].tolist()]
                bw = max(1.0, bx2 - bx1)
                bh = max(1.0, by2 - by1)
                cx = (bx1 + bx2) / 2.0
                cy = (by1 + by2) / 2.0
                raw_boxes.append({
                    "x1": bx1, "y1": by1, "x2": bx2, "y2": by2,
                    "cx": cx, "cy": cy, "w": bw, "h": bh,
                    "name": name, "conf": conf
                })

            if not raw_boxes:
                return None, 0, ""

            # 1. Filter outlier vertikal (digit STA berada pada satu baris horizontal di layar GPS)
            if len(raw_boxes) >= 3:
                med_cy = float(np.median([b["cy"] for b in raw_boxes]))
                med_h = float(np.median([b["h"] for b in raw_boxes]))
                filtered_v = [b for b in raw_boxes if abs(b["cy"] - med_cy) <= max(18.0, 0.9 * med_h)]
                if filtered_v:
                    raw_boxes = filtered_v

            # 2. Deduplikasi slot horizontal menggunakan 1D Horizontal IoMin (prioritas confidence tertinggi)
            raw_boxes.sort(key=lambda b: b["conf"], reverse=True)
            kept = []
            for b in raw_boxes:
                is_dup = False
                for k in kept:
                    inter_x = max(0.0, min(b["x2"], k["x2"]) - max(b["x1"], k["x1"]))
                    min_w = max(1.0, min(b["w"], k["w"]))
                    if (inter_x / min_w) > 0.45:
                        is_dup = True
                        break
                if not is_dup:
                    kept.append(b)

            # 3. Urutkan digit dari kiri ke kanan berdasarkan koordinat tengah X (cx)
            kept.sort(key=lambda b: b["cx"])

            # 4. Rangkai karakter digit menjadi angka STA
            chars = []
            for k in kept:
                nm = k["name"]
                if nm == "dot":
                    if "." not in chars:
                        chars.append(".")
                elif nm.isdigit():
                    chars.append(nm)

            sta_str = "".join(chars).strip(".")
            if not sta_str:
                return None, len(kept), ""

            val = float(sta_str)
            return val, len(kept), sta_str

        except Exception as e:
            logger.debug(f"Gagal membaca digit STA dari frame: {e}")
            return None, 0, ""

    @staticmethod
    def refine_sta_readings(
        times: List[float],
        raw_vals: List[Optional[float]],
        default_speed_mps: float = 8.33
    ) -> Tuple[List[float], List[float], List[float], float]:
        """
        Memurnikan sekuens pembacaan STA mentah di sepanjang video menggunakan
        Weighted Scale-Aware RANSAC & Monotonic Pairwise Filtering:
        - Mengatasi kehilangan digit nol di akhir pada STA >= 1000m (misal 780 -> 7800m).
        - Mengatasi kelebihan digit nol pada STA < 1000m (misal 5770 -> 577m).
        - Membuang spike/outlier tunggal dan menginterpolasi secara mulus dari STA Awal ke STA Akhir.
        Mengembalikan: (cleaned_sta_list, anchor_times, anchor_stas, global_speed_mps)
        """
        n = len(times)
        if n == 0:
            return [], [0.0], [0.0], default_speed_mps

        valid = [
            (i, float(times[i]), float(raw_vals[i]))
            for i in range(n)
            if raw_vals[i] is not None and raw_vals[i] >= 0
        ]

        if not valid:
            fallback = [round(t * default_speed_mps, 1) for t in times]
            return fallback, [0.0, max(1.0, times[-1])], [0.0, fallback[-1]], default_speed_mps

        if len(valid) == 1:
            _, t0, v0 = valid[0]
            res = [round(max(0.0, v0 + (t - t0) * default_speed_mps), 1) for t in times]
            return res, [times[0], times[-1]], [res[0], res[-1]], default_speed_mps

        # Bangun kandidat skala untuk setiap titik valid:
        # Kandidat asli diberi bobot 1.0; kandidat koreksi skala (*10, /10, *100) diberi bobot 0.65
        cands = []
        for idx, t, v in valid:
            pt_cands = [(v, 1.0)]
            if v >= 100 and v % 10 == 0:
                pt_cands.append((v / 10.0, 0.65))
            if 10 <= v < 1000:
                pt_cands.append((v * 10.0, 0.65))
            if 1 <= v < 100:
                pt_cands.append((v * 100.0, 0.50))
            cands.append((idx, t, pt_cands))

        m = len(cands)
        step = max(1, m // 32)
        best_score = -1.0
        best_model = None

        for i in range(0, m, step):
            _, t1, c1_list = cands[i]
            for j in range(i + 1, m, step):
                _, t2, c2_list = cands[j]
                dt = t2 - t1
                if dt < 0.2:
                    continue
                for v1, _ in c1_list:
                    for v2, _ in c2_list:
                        speed = (v2 - v1) / dt
                        # Kecepatan fisik kendaraan survei: 0.4 m/s (1.5 km/h) s/d 33.0 m/s (118 km/h)
                        if not (0.4 <= speed <= 33.0):
                            continue
                        score = 0.0
                        inliers = []
                        for idx_k, tk, ck_list in cands:
                            pred_k = v1 + speed * (tk - t1)
                            if pred_k < -20.0:
                                continue
                            tol = max(38.0, 0.12 * max(100.0, pred_k), 2.5 * (abs(tk - t1) ** 0.55))
                            best_c = None
                            best_w = 0.0
                            best_err = tol + 1.0
                            for cv, cw in ck_list:
                                err = abs(cv - pred_k)
                                if err <= tol and (err / cw) < best_err:
                                    best_err = err / cw
                                    best_c = cv
                                    best_w = cw
                            if best_c is not None:
                                score += best_w
                                inliers.append((idx_k, tk, best_c, best_w))
                        if score > best_score:
                            best_score = score
                            best_model = (t1, v1, speed, inliers)

        if not best_model or not best_model[3]:
            _, t0, v0 = valid[0]
            res = [round(max(0.0, v0 + (t - t0) * default_speed_mps), 1) for t in times]
            return res, [times[0], times[-1]], [res[0], res[-1]], default_speed_mps

        t1, v1, global_speed, inliers = best_model

        # Buang inlier awal yang lebih tinggi dari 2 inlier berikutnya (misal 590 sebelum 577, 596)
        pruned_inliers = []
        for idx_pos, item in enumerate(inliers):
            idx_k, tk, vk, wk = item
            next_items = inliers[idx_pos + 1: idx_pos + 4]
            if len(next_items) >= 2 and all(vk > nx[2] + 2.0 for nx in next_items[:2]):
                continue
            pruned_inliers.append((idx_k, tk, vk))

        if not pruned_inliers:
            pruned_inliers = [(x[0], x[1], x[2]) for x in inliers]

        # Filter konsistensi langkah berurutan (pairwise step filter)
        clean_pts: List[Tuple[int, float, float]] = []
        for idx_k, tk, vk in pruned_inliers:
            if not clean_pts:
                clean_pts.append((idx_k, tk, vk))
            else:
                prev_idx, prev_t, prev_v = clean_pts[-1]
                dt_step = max(0.1, tk - prev_t)
                dv_step = vk - prev_v
                if -5.0 <= dv_step <= (35.0 * dt_step + 25.0):
                    clean_pts.append((idx_k, tk, max(prev_v, vk)))
                elif len(clean_pts) == 1:
                    pred_prev = v1 + global_speed * (prev_t - t1)
                    pred_curr = v1 + global_speed * (tk - t1)
                    if abs(vk - pred_curr) < abs(prev_v - pred_prev):
                        clean_pts[0] = (idx_k, tk, vk)

        pt_times = [p[1] for p in clean_pts]
        pt_vals = [p[2] for p in clean_pts]

        # Hitung kecepatan lokal di awal dan akhir untuk ekstrapolasi batas ujung yang presisi
        if len(clean_pts) >= 2:
            dt_head = max(0.5, pt_times[min(3, len(pt_times) - 1)] - pt_times[0])
            dv_head = pt_vals[min(3, len(pt_vals) - 1)] - pt_vals[0]
            head_speed = dv_head / dt_head if 0.5 <= (dv_head / dt_head) <= 32.0 else global_speed

            dt_tail = max(0.5, pt_times[-1] - pt_times[max(0, len(pt_times) - 4)])
            dv_tail = pt_vals[-1] - pt_vals[max(0, len(pt_vals) - 4)]
            tail_speed = dv_tail / dt_tail if 0.5 <= (dv_tail / dt_tail) <= 32.0 else global_speed
        else:
            head_speed = global_speed
            tail_speed = global_speed

        pt_times_arr = np.array(pt_times, dtype=np.float64)
        pt_vals_arr = np.array(pt_vals, dtype=np.float64)

        result: List[float] = []
        for t in times:
            if t < pt_times_arr[0]:
                val = max(0.0, float(pt_vals_arr[0]) - head_speed * (float(pt_times_arr[0]) - t))
            elif t > pt_times_arr[-1]:
                val = float(pt_vals_arr[-1]) + tail_speed * (t - float(pt_times_arr[-1]))
            else:
                val = float(np.interp(t, pt_times_arr, pt_vals_arr))
            result.append(round(val, 1))

        # Pastikan urutan STA monotonik tidak turun
        for k in range(1, len(result)):
            if result[k] < result[k - 1]:
                result[k] = result[k - 1]

        return result, pt_times, pt_vals, float(global_speed)

    def precalibrate_from_video(
        self,
        cap: cv2.VideoCapture,
        fps: float,
        total_frames: int,
        num_samples: int = 28
    ) -> None:
        """
        Melakukan pre-scanning cepat (~25 frame menyebar dari awal hingga akhir video)
        menggunakan model gps_sta_model/weights/best.pt pada GPU agar nilai STA Awal,
        STA Akhir, dan kurva referensi kecepatan diketahui secara presisi sejak Frame 0.
        """
        self.reset()
        if self.model is None or not self.is_digit_model or total_frames <= 0:
            return

        fps_val = max(1.0, fps)
        # Pilih indeks sampel yang mencakup awal jalan, tengah jalan, dan akhir jalan
        early_indices = [0, 15, 30, 60, 120, 240]
        late_indices = [max(0, total_frames - 1 - d) for d in [240, 120, 60, 30, 15, 1]]
        mid_indices = np.linspace(0, max(0, total_frames - 1), num=max(6, num_samples - 10), dtype=int).tolist()
        sample_indices = sorted(list(set(
            idx for idx in (early_indices + mid_indices + late_indices)
            if 0 <= idx < total_frames
        )))

        sample_times: List[float] = []
        sample_raws: List[Optional[float]] = []

        for fidx in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue
            t_sec = fidx / fps_val
            raw_val, det_cnt, _ = self._read_raw_sta_from_frame(frame)
            sample_times.append(t_sec)
            sample_raws.append(raw_val)
            if raw_val is not None:
                self._raw_history.append((fidx, t_sec, raw_val))

        # Kembalikan posisi video ke Frame 0 untuk pemrosesan utama
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        valid_cnt = sum(1 for v in sample_raws if v is not None)
        if valid_cnt >= 1:
            cleaned, anchor_t, anchor_v, g_speed = self.refine_sta_readings(
                sample_times, sample_raws, default_speed_mps=self._est_speed_mps
            )
            self._anchor_times = np.array(sample_times, dtype=np.float64)
            self._anchor_stas = np.array(cleaned, dtype=np.float64)
            self._global_speed_mps = g_speed
            self._est_speed_mps = g_speed
            self._detected_start_sta = cleaned[0]
            self._last_known_sta = cleaned[0]
            logger.info(
                f"🎯 Pre-kalibrasi STA Model Selesai: STA Awal={cleaned[0]:.1f}m -> "
                f"STA Akhir={cleaned[-1]:.1f}m (Valid sampel: {valid_cnt}/{len(sample_indices)})"
            )

    def predict(
        self,
        frame: Optional[np.ndarray] = None,
        frame_idx: int = 0,
        fps: float = 30.0,
        current_time_s: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Menghasilkan pembacaan STA (meter) dan koordinat GPS pada frame saat ini
        menggunakan model gps_sta_model/weights/best.pt + ROI + pelacakan trajektori.
        """
        fps_val = max(1.0, fps)
        t_sec = current_time_s if current_time_s is not None else (frame_idx / fps_val)

        raw_sta: Optional[float] = None
        detected_features_count = self._last_features_count

        if self.model is not None and frame is not None:
            raw_sta, detected_features_count, raw_str = self._read_raw_sta_from_frame(frame)
            self._last_features_count = detected_features_count
            if raw_str:
                self._last_raw_sta_str = raw_str
            if raw_sta is not None:
                self._raw_history.append((frame_idx, t_sec, raw_sta))

        # Tentukan nilai STA terbaik pada waktu t_sec
        if self._anchor_times is not None and self._anchor_stas is not None and len(self._anchor_times) >= 2:
            # Interpolasi dari kurva global hasil pre-kalibrasi
            if t_sec <= self._anchor_times[0]:
                expected_sta = float(self._anchor_stas[0])
            elif t_sec >= self._anchor_times[-1]:
                expected_sta = float(self._anchor_stas[-1]) + (self._global_speed_mps or self._est_speed_mps) * (t_sec - float(self._anchor_times[-1]))
            else:
                expected_sta = float(np.interp(t_sec, self._anchor_times, self._anchor_stas))

            chosen_sta = expected_sta
            if raw_sta is not None and raw_sta >= 0:
                # Cek kandidat skala dari raw_sta yang cocok dengan kurva referensi
                candidates = [raw_sta]
                if raw_sta >= 100 and raw_sta % 10 == 0:
                    candidates.append(raw_sta / 10.0)
                if 10 <= raw_sta < 1000:
                    candidates.append(raw_sta * 10.0)
                if 1 <= raw_sta < 100:
                    candidates.append(raw_sta * 100.0)

                tol = max(35.0, 0.10 * max(100.0, expected_sta))
                best_cand = min(candidates, key=lambda c: abs(c - expected_sta))
                if abs(best_cand - expected_sta) <= tol:
                    # Gunakan rata-rata berbobot antara pembacaan langsung frame & kurva agar responsif dan stabil
                    chosen_sta = 0.65 * best_cand + 0.35 * expected_sta

            if self._last_known_sta is not None:
                current_sta = round(max(self._last_known_sta, chosen_sta), 1)
            else:
                current_sta = round(max(0.0, chosen_sta), 1)
        else:
            # Mode tanpa pre-kalibrasi (online / single frame)
            dt = max(0.0, t_sec - self._last_known_time_s)
            if self._last_known_sta is None:
                if raw_sta is not None:
                    current_sta = round(raw_sta, 1)
                    self._detected_start_sta = current_sta
                else:
                    current_sta = round(self.sta_start_m + t_sec * self._est_speed_mps, 1)
            else:
                expected_sta = self._last_known_sta + self._est_speed_mps * dt
                if raw_sta is not None:
                    candidates = [raw_sta]
                    if raw_sta >= 100 and raw_sta % 10 == 0:
                        candidates.append(raw_sta / 10.0)
                    if 10 <= raw_sta < 1000:
                        candidates.append(raw_sta * 10.0)
                    tol = max(45.0, 35.0 * dt + 20.0)
                    best_cand = min(candidates, key=lambda c: abs(c - expected_sta))
                    if abs(best_cand - expected_sta) <= tol and best_cand >= self._last_known_sta - 5.0:
                        current_sta = round(max(self._last_known_sta, best_cand), 1)
                    else:
                        current_sta = round(expected_sta, 1)
                else:
                    current_sta = round(expected_sta, 1)

        if self._detected_start_sta is None:
            self._detected_start_sta = current_sta

        # Perhitungan koordinat GPS relatif terhadap jarak tempuh dari awal ruas
        dist_from_start = max(0.0, current_sta - (self._detected_start_sta or 0.0))
        lat = self.gps_start_coord[0] + (dist_from_start * 0.000008)
        lon = self.gps_start_coord[1] + (dist_from_start * 0.000008)

        self._last_known_sta = current_sta
        self._last_known_time_s = t_sec
        self._last_known_lat = lat
        self._last_known_lon = lon

        return {
            "sta_m": current_sta,
            "raw_sta_m": raw_sta,
            "lat": round(lat, 6),
            "lon": round(lon, 6),
            "model_features": detected_features_count,
            "source": f"gps_sta_model ({os.path.basename(self.model_loaded_path) if self.model_loaded_path else 'best.pt'})"
        }

    def refine_frame_records(
        self,
        frame_records: List[Any],
        preview_frames: Optional[List[Dict[str, Any]]] = None,
        fps: float = 30.0
    ) -> None:
        """
        Memurnikan nilai STA pada seluruh FrameRecord setelah pembacaan video selesai,
        memastikan STA Awal dan STA Akhir seluruh segmen jalan sesuai dengan
        pembacaan model gps_sta_model/weights/best.pt di sepanjang video.
        """
        if not frame_records or not self._raw_history:
            return

        fps_val = max(1.0, fps)
        # Gabungkan seluruh pembacaan raw berdasarkan frame_idx unik
        raw_by_fidx: Dict[int, Tuple[float, float]] = {}
        for fidx, t_sec, r_val in self._raw_history:
            if r_val is not None:
                raw_by_fidx[fidx] = (t_sec, r_val)

        if not raw_by_fidx:
            return

        # Susun daftar waktu untuk seluruh frame_records + titik raw_by_fidx
        all_fidxs = sorted(list(set([f.frame_idx for f in frame_records] + list(raw_by_fidx.keys()))))
        all_times = [idx / fps_val for idx in all_fidxs]
        all_raws = [raw_by_fidx[idx][1] if idx in raw_by_fidx else None for idx in all_fidxs]

        cleaned_stas, _, _, _ = self.refine_sta_readings(
            all_times, all_raws, default_speed_mps=self._est_speed_mps
        )

        fidx_to_sta = dict(zip(all_fidxs, cleaned_stas))
        start_sta = cleaned_stas[0] if cleaned_stas else 0.0

        for rec in frame_records:
            if rec.frame_idx in fidx_to_sta:
                rec.sta_m = fidx_to_sta[rec.frame_idx]
                dist = max(0.0, rec.sta_m - start_sta)
                rec.lat = round(self.gps_start_coord[0] + (dist * 0.000008), 6)
                rec.lon = round(self.gps_start_coord[1] + (dist * 0.000008), 6)

        if preview_frames:
            for pf in preview_frames:
                fidx = pf.get("frame_idx")
                if fidx in fidx_to_sta:
                    pf["sta_m"] = fidx_to_sta[fidx]
