"""
Adapter Deteksi Kerusakan Jalan (Road Damage Detector).
Mendukung kalibrasi otomatis adaptif untuk luas retak dan lebar retak.
Antarmuka seragam: RoadDamageDetector.predict(frame) -> list[{class, conf, bbox/mask, area_px, width_px, sdi_category}]
"""
import os
import logging
from typing import List, Dict, Any, Optional
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

logger = logging.getLogger(__name__)


class RoadDamageDetector:
    """
    Adapter model YOLO Deteksi Kerusakan Jalan (RDD2022).
    Mendeteksi kerusakan (retak, lubang, dll) dengan kalibrasi otomatis.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        imgsz: int = 640,
        device: str = "cpu",
        class_mapping: Optional[Dict[str, str]] = None,
        meter_per_pixel: float = 0.005
    ):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.imgsz = imgsz
        self.device = device
        self.meter_per_pixel = meter_per_pixel

        # Faktor luas adaptif per kelas
        self.crack_area_factors = {
            "alligator crack": 0.35,
            "longitudinal crack": 0.15,
            "transverse crack": 0.15,
            "other corruption": 0.25,
            "Pothole": 0.785,
        }

        # Mapping kelas deteksi ke kategori SDI
        self.class_mapping = class_mapping or {
            "longitudinal crack": "crack",
            "transverse crack": "crack",
            "alligator crack": "crack",
            "other corruption": "other",
            "Pothole": "pothole",
        }

        candidates = [
            model_path,
            os.path.join("road_damage_models", "runs", "detect", "rdd_baseline", "run1", "weights", "best.pt"),
            os.path.join("road_damage_models", "weights", "yolo26n.pt"),
        ]

        self.model = None
        self.model_loaded_path = None

        if YOLO is None:
            logger.warning("Library ultralytics belum terpasang. Mode mock aktif.")
            return

        for p in candidates:
            if p and os.path.exists(p):
                try:
                    self.model = YOLO(p)
                    self.model_loaded_path = p
                    logger.info(f"Berhasil memuat model RoadDamageDetector dari: {p}")
                    break
                except Exception as e:
                    logger.warning(f"Gagal memuat model dari {p}: {e}")

        if self.model is None:
            logger.warning("Tidak ditemukan checkpoint model road_damage yang valid.")

    def update_auto_calibration(
        self,
        meter_per_pixel: float,
        crack_area_factors: Optional[Dict[str, float]] = None
    ) -> None:
        """Memperbarui kalibrasi secara otomatis dari dimensi frame dan lebar jalan."""
        self.meter_per_pixel = meter_per_pixel
        if crack_area_factors:
            self.crack_area_factors.update(crack_area_factors)

    def predict(
        self,
        frame: np.ndarray,
        conf_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Melakukan inferensi deteksi kerusakan pada satu frame.
        """
        conf = conf_threshold if conf_threshold is not None else self.conf_threshold

        if self.model is None:
            return []

        results = self.model.predict(
            source=frame,
            conf=conf,
            iou=self.iou_threshold,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False
        )

        detections: List[Dict[str, Any]] = []
        if not results or len(results) == 0:
            return detections

        r = results[0]
        boxes = getattr(r, "boxes", None)
        masks = getattr(r, "masks", None)

        if boxes is None or len(boxes) == 0:
            return detections

        has_mask_data = masks is not None and getattr(masks, "data", None) is not None

        for i, box in enumerate(boxes):
            cls_id = int(box.cls[0].item())
            class_name = self.model.names.get(cls_id, f"class_{cls_id}")
            score = float(box.conf[0].item())
            xyxy = [float(c) for c in box.xyxy[0].tolist()]
            x1, y1, x2, y2 = xyxy
            w_box = max(0.0, x2 - x1)
            h_box = max(0.0, y2 - y1)
            box_area = w_box * h_box

            sdi_cat = self.class_mapping.get(class_name, "other")
            mask_arr = None

            if has_mask_data and i < len(masks.data):
                # Segmentasi mask riil
                mask_tensor = masks.data[i].cpu().numpy()
                mask_arr = mask_tensor > 0.5
                area_px = float(np.sum(mask_arr))
                width_px = min(w_box, h_box)
            else:
                # Estimasi adaptif otomatis berdasarkan jenis kerusakan
                factor = self.crack_area_factors.get(class_name, 0.20)
                area_px = box_area * factor
                if sdi_cat == "crack":
                    width_px = max(1.0, min(w_box, h_box) * 0.15)
                else:
                    width_px = min(w_box, h_box)

            # Konversi fisik otomatis menggunakan rasio meter_per_pixel
            area_m2 = area_px * (self.meter_per_pixel ** 2)
            width_mm = width_px * self.meter_per_pixel * 1000.0

            detections.append({
                "class": class_name,
                "sdi_category": sdi_cat,
                "conf": round(score, 3),
                "bbox": [round(c, 1) for c in xyxy],
                "mask": mask_arr,
                "area_px": round(area_px, 1),
                "area_m2": round(area_m2, 4),
                "width_px": round(width_px, 1),
                "width_mm": round(width_mm, 2),
            })

        return detections
