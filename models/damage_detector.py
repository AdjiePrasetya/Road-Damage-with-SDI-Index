"""
Adapter Deteksi & Segmentasi Kerusakan Jalan (Multi-Model SDI PKRMS).
Memuat 3 model YOLO Instance Segmentation spesifik sesuai panduan SDI:
1. Cracks (crack_lt_1mm, crack_1_3mm, crack_gt_3mm)
2. Potholes (pothole)
3. Rutting (rut_lt_1cm, rut_1_3cm, rut_gt_3cm)
"""
import os
import cv2
import logging
from typing import List, Dict, Any, Optional
import numpy as np
from core.device import get_optimal_device

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

logger = logging.getLogger(__name__)


class RoadDamageDetector:
    """
    Adapter terpadu untuk mengeksekusi 3 model segmentasi kerusakan jalan
    (Cracks, Potholes, Rutting) dengan interface seragam.
    """

    def __init__(
        self,
        cracks_model_path: Optional[str] = None,
        potholes_model_path: Optional[str] = None,
        rutting_model_path: Optional[str] = None,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        imgsz: int = 640,
        device: str = "auto",
        meter_per_pixel: float = 0.005,
        class_mapping: Optional[Dict[str, str]] = None,
        **kwargs
    ):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.imgsz = imgsz
        self.meter_per_pixel = meter_per_pixel

        # Deteksi & prioritaskan GPU NVIDIA secara otomatis
        self.device, self.device_info = get_optimal_device(device)

        self.class_mapping = class_mapping or {
            "crack_lt_1mm": "crack",
            "crack_1_3mm": "crack",
            "crack_gt_3mm": "crack",
            "pothole": "pothole",
            "rut_lt_1cm": "rutting",
            "rut_1_3cm": "rutting",
            "rut_gt_3cm": "rutting",
        }

        # Jalur default masing-masing model
        p_cracks = cracks_model_path or os.path.join("road_damage_models", "cracks", "weights", "best.pt")
        p_potholes = potholes_model_path or os.path.join("road_damage_models", "potholes", "weights", "best.pt")
        p_rutting = rutting_model_path or os.path.join("road_damage_models", "rutting", "weights", "best.pt")

        self.cracks_model = None
        self.potholes_model = None
        self.rutting_model = None

        if YOLO is not None:
            # 1. Model Cracks
            if os.path.exists(p_cracks):
                try:
                    self.cracks_model = YOLO(p_cracks)
                    logger.info(f"Loaded Cracks model: {p_cracks}")
                except Exception as e:
                    logger.warning(f"Failed to load Cracks model from {p_cracks}: {e}")

            # 2. Model Potholes
            if os.path.exists(p_potholes):
                try:
                    self.potholes_model = YOLO(p_potholes)
                    logger.info(f"Loaded Potholes model: {p_potholes}")
                except Exception as e:
                    logger.warning(f"Failed to load Potholes model from {p_potholes}: {e}")

            # 3. Model Rutting
            if os.path.exists(p_rutting):
                try:
                    self.rutting_model = YOLO(p_rutting)
                    logger.info(f"Loaded Rutting model: {p_rutting}")
                except Exception as e:
                    logger.warning(f"Failed to load Rutting model from {p_rutting}: {e}")

    def update_auto_calibration(
        self,
        meter_per_pixel: float,
        crack_area_factors: Optional[Dict[str, float]] = None
    ) -> None:
        """Memperbarui kalibrasi meter per pixel dari dimensi frame video."""
        self.meter_per_pixel = meter_per_pixel

    def _extract_from_yolo_result(
        self,
        results: Any,
        default_category: str
    ) -> List[Dict[str, Any]]:
        """Mengekstrak bounding box, mask polygon, dan luas metrik dari hasil prediksi YOLO."""
        extracted: List[Dict[str, Any]] = []
        if not results or len(results) == 0:
            return extracted

        r = results[0]
        boxes = getattr(r, "boxes", None)
        masks = getattr(r, "masks", None)

        if boxes is None or len(boxes) == 0:
            return extracted

        has_mask_xy = masks is not None and getattr(masks, "xy", None) is not None
        has_mask_data = masks is not None and getattr(masks, "data", None) is not None

        for i, box in enumerate(boxes):
            cls_id = int(box.cls[0].item())
            class_name = r.names.get(cls_id, f"class_{cls_id}")
            score = float(box.conf[0].item())
            xyxy = [float(c) for c in box.xyxy[0].tolist()]
            x1, y1, x2, y2 = xyxy
            w_box = max(0.0, x2 - x1)
            h_box = max(0.0, y2 - y1)
            box_area = w_box * h_box

            sdi_cat = self.class_mapping.get(class_name, default_category)
            polygon = None
            area_px = 0.0

            if has_mask_xy and i < len(masks.xy) and len(masks.xy[i]) >= 3:
                poly_arr = masks.xy[i]
                if isinstance(poly_arr, np.ndarray):
                    polygon = poly_arr.tolist()
                    area_px = float(cv2.contourArea(poly_arr.astype(np.float32)))
                else:
                    polygon = list(poly_arr)
                    area_px = float(cv2.contourArea(np.array(poly_arr, dtype=np.float32)))
                if area_px <= 0:
                    area_px = box_area * 0.40
            elif has_mask_data and i < len(masks.data):
                mask_tensor = masks.data[i].cpu().numpy()
                mask_arr = mask_tensor > 0.5
                area_px = float(np.sum(mask_arr))
            else:
                area_px = box_area * 0.35

            area_m2 = area_px * (self.meter_per_pixel ** 2)

            # Estimasi lebar retak (mm) berbasis kelas prediksi model
            width_mm = 0.0
            if "crack_gt_3mm" in class_name:
                width_mm = 4.0
            elif "crack_1_3mm" in class_name:
                width_mm = 2.0
            elif "crack_lt_1mm" in class_name:
                width_mm = 0.8

            # Estimasi kedalaman alur (cm) berbasis kelas prediksi model
            rut_depth_cm = 0.0
            if "rut_gt_3cm" in class_name:
                rut_depth_cm = 4.0
            elif "rut_1_3cm" in class_name:
                rut_depth_cm = 2.0
            elif "rut_lt_1cm" in class_name:
                rut_depth_cm = 0.8

            extracted.append({
                "class": class_name,
                "sdi_category": sdi_cat,
                "conf": round(score, 3),
                "bbox": [round(c, 1) for c in xyxy],
                "polygon": polygon,
                "area_px": round(area_px, 1),
                "area_m2": round(area_m2, 4),
                "width_mm": width_mm,
                "rut_depth_cm": rut_depth_cm,
            })

        return extracted

    def predict(
        self,
        frame: np.ndarray,
        conf_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Menjalankan ketiga model segmentasi (cracks, potholes, rutting) pada satu frame.
        Mengembalikan list gabungan deteksi kerusakan.
        """
        conf = conf_threshold if conf_threshold is not None else self.conf_threshold
        all_detections: List[Dict[str, Any]] = []

        # 1. Inferensi Model Cracks
        if self.cracks_model is not None:
            try:
                res_cracks = self.cracks_model.predict(
                    source=frame,
                    conf=conf,
                    iou=self.iou_threshold,
                    imgsz=self.imgsz,
                    device=self.device,
                    verbose=False
                )
                all_detections.extend(self._extract_from_yolo_result(res_cracks, default_category="crack"))
            except Exception as e:
                logger.debug(f"Cracks inference error: {e}")

        # 2. Inferensi Model Potholes
        if self.potholes_model is not None:
            try:
                res_potholes = self.potholes_model.predict(
                    source=frame,
                    conf=conf,
                    iou=self.iou_threshold,
                    imgsz=self.imgsz,
                    device=self.device,
                    verbose=False
                )
                all_detections.extend(self._extract_from_yolo_result(res_potholes, default_category="pothole"))
            except Exception as e:
                logger.debug(f"Potholes inference error: {e}")

        # 3. Inferensi Model Rutting
        if self.rutting_model is not None:
            try:
                res_rutting = self.rutting_model.predict(
                    source=frame,
                    conf=conf,
                    iou=self.iou_threshold,
                    imgsz=self.imgsz,
                    device=self.device,
                    verbose=False
                )
                all_detections.extend(self._extract_from_yolo_result(res_rutting, default_category="rutting"))
            except Exception as e:
                logger.debug(f"Rutting inference error: {e}")

        return all_detections
