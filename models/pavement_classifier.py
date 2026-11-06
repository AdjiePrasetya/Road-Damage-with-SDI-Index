"""
Adapter Klasifikasi Jenis Perkerasan Jalan (Pavement Classifier).
Antarmuka seragam: PavementClassifier.predict(frame) -> {label, conf}
"""
import os
import logging
from typing import Dict, Any, Optional
import numpy as np
from core.device import get_optimal_device

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

logger = logging.getLogger(__name__)


class PavementClassifier:
    """
    Adapter model YOLO Klasifikasi Jenis Perkerasan Jalan.
    Mengklasifikasikan permukaan jalan ke: Hotmix, Lapen, Beton, Kerikil, Tanah.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        weights_path: Optional[str] = None,
        imgsz: int = 320,
        device: str = "auto",
        class_mapping: Optional[Dict[str, str]] = None,
        **kwargs
    ):
        self.imgsz = imgsz
        self.device, self.device_info = get_optimal_device(device)

        # Mapping label internal model ke label tampilan PKRMS
        self.class_mapping = class_mapping or {
            "hotmix": "Hotmix",
            "lapen": "Lapen",
            "perkerasan_beton": "Beton",
            "kerikil": "Kerikil",
            "tanah_belum_tembus": "Tanah",
        }

        target_path = model_path or weights_path
        candidates = [
            target_path,
            os.path.join("perkerasan_clf_models", "runs", "perkerasan_cls", "weights", "best.pt"),
            os.path.join("perkerasan_clf_models", "runs", "weights", "best.pt"),
            os.path.join("perkerasan_clf_models", "weights", "best.pt"),
            os.path.join("perkerasan_clf_models", "weights", "yolo26n.pt"),
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
                    logger.info(f"Berhasil memuat model PavementClassifier dari: {p}")
                    break
                except Exception as e:
                    logger.warning(f"Gagal memuat model dari {p}: {e}")

        if self.model is None:
            logger.warning("Tidak ditemukan checkpoint model perkerasan_clf yang valid.")

    def predict(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Melakukan klasifikasi jenis perkerasan pada satu frame.
        Mengembalikan:
          {
            'label': str (Label terstandar PKRMS: Hotmix, Beton, Lapen, dst),
            'raw_label': str (Label asli dari model),
            'conf': float (Probabilitas tertinggi 0.0 - 1.0),
            'all_probs': Dict[str, float] (Distribusi probabilitas semua kelas)
          }
        """
        if self.model is None:
            # Fallback default
            return {
                "label": "Hotmix",
                "raw_label": "hotmix",
                "conf": 1.0,
                "all_probs": {"hotmix": 1.0}
            }

        try:
            results = self.model.predict(
                source=frame,
                imgsz=self.imgsz,
                device=self.device,
                verbose=False
            )

            if not results or len(results) == 0:
                return {"label": "Hotmix", "raw_label": "unknown", "conf": 0.0, "all_probs": {}}

            r = results[0]
            probs_obj = getattr(r, "probs", None)

            if probs_obj is not None:
                top1_idx = int(probs_obj.top1)
                top1_conf = float(probs_obj.top1conf.item())
                raw_name = self.model.names.get(top1_idx, f"class_{top1_idx}")

                all_probs = {}
                if hasattr(probs_obj, "data") and probs_obj.data is not None:
                    p_data = probs_obj.data.cpu().numpy()
                    for idx, p_val in enumerate(p_data):
                        c_name = self.model.names.get(idx, f"class_{idx}")
                        all_probs[c_name] = round(float(p_val), 4)

                mapped_label = self.class_mapping.get(raw_name, raw_name.title())

                return {
                    "label": mapped_label,
                    "raw_label": raw_name,
                    "conf": round(top1_conf, 3),
                    "all_probs": all_probs
                }
            else:
                # Jika checkpoint bukan task classify (misal detection model)
                return {"label": "Hotmix", "raw_label": "default", "conf": 0.9, "all_probs": {}}

        except Exception as e:
            logger.error(f"Error pada PavementClassifier.predict: {e}")
            return {"label": "Hotmix", "raw_label": "error", "conf": 0.0, "all_probs": {}}

