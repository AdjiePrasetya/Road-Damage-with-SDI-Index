"""
Adapter Estimasi STA dan Posisi GPS (StaEstimator).
Antarmuka seragam: StaEstimator.predict(frame/video, frame_idx, fps) -> {sta_m, lat, lon}
"""
import os
import re
import logging
from typing import Dict, Any, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)

try:
    import easyocr
except ImportError:
    easyocr = None


class StaEstimator:
    """
    Adapter untuk memperoleh posisi STA (Stationing / jarak kumulatif dalam meter)
    dan koordinat GPS (latitude, longitude) dari setiap frame video survei.
    
    Mendukung 3 mode:
      1. 'speed_time'   : Berdasarkan kecepatan rata-rata survei (km/jam) dan waktu frame.
      2. 'ocr_overlay'  : Membaca teks HUD / overlay video (koordinat / STA) via OCR.
      3. 'custom_yolo'  : Menggunakan model deteksi patok KM / angka (placeholder TODO).
    """

    def __init__(
        self,
        mode: str = "speed_time",
        sta_start_m: float = 0.0,
        default_speed_kmh: float = 30.0,
        ocr_crop_box: Optional[Tuple[float, float, float, float]] = None,
        custom_model_path: Optional[str] = None,
        gps_start_coord: Optional[Tuple[float, float]] = None,
        gps_end_coord: Optional[Tuple[float, float]] = None
    ):
        self.mode = mode
        self.sta_start_m = sta_start_m
        self.default_speed_kmh = default_speed_kmh
        # Box crop normalisasi [ymin, xmin, ymax, xmax] untuk OCR overlay
        self.ocr_crop_box = ocr_crop_box or (0.85, 0.0, 1.0, 0.50)
        self.custom_model_path = custom_model_path
        self.gps_start_coord = gps_start_coord or (-6.2088, 106.8456) # Default Jakarta / Jawa Barat
        self.gps_end_coord = gps_end_coord

        self.ocr_reader = None
        if self.mode == "ocr_overlay" and easyocr is not None:
            try:
                self.ocr_reader = easyocr.Reader(["en"], gpu=False)
                logger.info("EasyOCR reader berhasil diinisialisasi untuk StaEstimator.")
            except Exception as e:
                logger.warning(f"Gagal menginisialisasi EasyOCR: {e}. Fallback ke speed_time.")
                self.mode = "speed_time"

        # Cache STA terakhir untuk interpolasi jika OCR drop
        self._last_known_sta = self.sta_start_m
        self._last_known_lat = self.gps_start_coord[0]
        self._last_known_lon = self.gps_start_coord[1]

    def predict(
        self,
        frame: Optional[np.ndarray] = None,
        frame_idx: int = 0,
        fps: float = 30.0,
        current_time_s: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Menghasilkan perkiraan STA (jarak meter dari titik awal) dan koordinat GPS.
        Mengembalikan:
          {
            'sta_m': float,
            'lat': Optional[float],
            'lon': Optional[float],
            'source': str
          }
        """
        fps_val = max(1.0, fps)
        t_sec = current_time_s if current_time_s is not None else (frame_idx / fps_val)

        # ----------------------------------------------------------------------
        # MODE 1: SPEED & TIME (PALING STABIL & DIREKOMENDASIKAN UNTUK SURVEI)
        # ----------------------------------------------------------------------
        if self.mode == "speed_time" or frame is None:
            speed_mps = (self.default_speed_kmh * 1000.0) / 3600.0
            distance_traveled = t_sec * speed_mps
            current_sta = round(self.sta_start_m + distance_traveled, 1)

            # Estimasi pergeseran koordinat (asumsi arah survei ~ 0.00001 deg ~ 1.1 meter)
            lat = self.gps_start_coord[0] + (distance_traveled * 0.000008)
            lon = self.gps_start_coord[1] + (distance_traveled * 0.000008)

            self._last_known_sta = current_sta
            self._last_known_lat = lat
            self._last_known_lon = lon

            return {
                "sta_m": current_sta,
                "lat": round(lat, 6),
                "lon": round(lon, 6),
                "source": "speed_time"
            }

        # ----------------------------------------------------------------------
        # MODE 2: OCR OVERLAY (MEMBACA KOORDINAT / STA DARI VIDEO)
        # ----------------------------------------------------------------------
        elif self.mode == "ocr_overlay":
            if self.ocr_reader is not None and frame is not None:
                h, w = frame.shape[:2]
                ymin, xmin, ymax, xmax = self.ocr_crop_box
                crop = frame[int(ymin * h):int(ymax * h), int(xmin * w):int(xmax * w)]

                try:
                    results = self.ocr_reader.readtext(crop, detail=0)
                    text = " ".join(results)
                    # Pola regex pencarian STA: "STA 0+150" atau "STA 150" atau "KM 12"
                    sta_match = re.search(r'STA\s*(\d+)(?:\+(\d+))?', text, re.IGNORECASE)
                    if sta_match:
                        km = int(sta_match.group(1))
                        m = int(sta_match.group(2)) if sta_match.group(2) else 0
                        read_sta = float(km * 1000 + m) if sta_match.group(2) else float(km)
                        self._last_known_sta = read_sta
                        return {
                            "sta_m": read_sta,
                            "lat": self._last_known_lat,
                            "lon": self._last_known_lon,
                            "source": "ocr_success"
                        }
                except Exception as e:
                    logger.debug(f"OCR gagal membaca frame {frame_idx}: {e}")

            # Fallback ke interpolasi waktu jika OCR gagal pada frame ini
            speed_mps = (self.default_speed_kmh * 1000.0) / 3600.0
            distance_traveled = t_sec * speed_mps
            current_sta = round(self.sta_start_m + distance_traveled, 1)
            return {
                "sta_m": current_sta,
                "lat": self._last_known_lat,
                "lon": self._last_known_lon,
                "source": "ocr_fallback_speed"
            }

        # ----------------------------------------------------------------------
        # MODE 3: CUSTOM YOLO DETECTION (TODO USER)
        # ----------------------------------------------------------------------
        else:
            # TODO (USER): Jika Anda memiliki bobot model deteksi angka / patok kilometer
            # di folder gps_sta_model/, integrasikan pemanggilan model di sini.
            speed_mps = (self.default_speed_kmh * 1000.0) / 3600.0
            current_sta = round(self.sta_start_m + t_sec * speed_mps, 1)
            return {
                "sta_m": current_sta,
                "lat": self._last_known_lat,
                "lon": self._last_known_lon,
                "source": "custom_yolo_placeholder"
            }

