"""
Adapter Deteksi Kerusakan Jalan (Road Damage Detector).
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
    Mendeteksi jenis kerusakan (retak, lubang, dll) dan menghitung estimasi luas serta lebar.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        imgsz: int = 640,
        device: str = "cpu",
        class_mapping: Optional[Dict[str, str]] = None,
        bbox_crack_area_factor: float = 0.20,
        meter_per_pixel: float = 0.005
    ):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.imgsz = imgsz
        self.device = device
        self.bbox_crack_area_factor = bbox_crack_area_factor
        self.meter_per_pixel = meter_per_pixel

        # Mapping kelas deteksi ke kategori SDI
        self.class_mapping = class_mapping or {
            "longitudinal crack": "crack",
            "transverse crack": "crack",
            "alligator crack": "crack",
            "other corruption": "other",
            "Pothole": "pothole",
        }

        # Jalur default model
        candidates = [
            model_path,
            os.path.join("road_damage_models", "runs", "detect", "rdd_baseline", "run1", "weights", "best.pt"),
            os.path.join("road_damage_models", "weights", "yolo26n.pt"),
        ]

        self.model = None
        self.model_loaded_path = None
        self.has_masks = False

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

    def predict(
        self,
        frame: np.ndarray,
        conf_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Melakukan inferensi deteksi kerusakan pada satu frame.
        Mengembalikan list of detection dict:
          [{
            'class': str (nama kelas asli model),
            'sdi_category': str ('crack', 'pothole', 'other'),
            'conf': float,
            'bbox': [x1, y1, x2, y2],
            'mask': Optional[np.ndarray],
            'area_px': float (luas estimasi dalam pixel),
            'area_m2': float (luas dalam m2 menggunakan kalibrasi meter_per_pixel),
            'width_px': float (estimasi lebar dalam pixel),
            'width_mm': float (estimasi lebar dalam mm),
            'warning': Optional[str]
          }]
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

            # Perhitungan luas & lebar
            mask_arr = None
            warning_msg = None

            if has_mask_data and i < len(masks.data):
                # Model Instance Segmentation: gunakan polygon mask asli
                mask_tensor = masks.data[i].cpu().numpy()
                mask_arr = mask_tensor > 0.5
                area_px = float(np.sum(mask_arr))
                # Estimasi lebar dari perimeter / area
                width_px = min(w_box, h_box)
            else:
                # Model Bounding Box Deteksi:
                # Untuk crack, retak berbentuk garis sehingga tidak memenuhi seluruh bbox.
                # Gunakan faktor koreksi bbox_crack_area_factor
                if sdi_cat == "crack":
                    area_px = box_area * self.bbox_crack_area_factor
                    # Lebar retak diestimasi dari rasio ketebalan retak
                    width_px = max(1.0, min(w_box, h_box) * 0.15)
                    warning_msg = (
                        f"Area retak diestimasi {self.bbox_crack_area_factor*100:.0f}% "
                        "dari bounding box (model deteksi, bukan segmentasi mask)."
                    )
                elif sdi_cat == "pothole":
                    # Lubang biasanya elips/lingkaran (pi/4 ~ 0.785 area bbox)
                    area_px = box_area * 0.785
                    width_px = min(w_box, h_box)
                else:
                    area_px = box_area
                    width_px = min(w_box, h_box)

            # Konversi pixel ke unit fisik meter & milimeter
            # 1 pixel = meter_per_pixel meter
            # 1 pixel area = (meter_per_pixel)^2 m2
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
                "warning": warning_msg,
            })

        return detections

