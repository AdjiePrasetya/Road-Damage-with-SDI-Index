"""
Modul Manajemen Komputasi Hardware (GPU CUDA & CPU Multi-threading).
Memprioritaskan penggunaan GPU NVIDIA secara maksimal jika tersedia,
atau memaksimalkan seluruh core CPU (100% utilitas) jika berjalan tanpa GPU.
"""
import os
import torch
import cv2
import logging
from typing import Tuple, Dict, Any

logger = logging.getLogger(__name__)


def get_optimal_device(preferred: str = "auto") -> Tuple[str, Dict[str, Any]]:
    """
    Menentukan device komputasi terbaik:
    - Prioritas 1: NVIDIA GPU (CUDA) jika tersedia -> 'cuda:0'.
      Mengaktifkan cuDNN benchmark dan optimasi VRAM.
    - Prioritas 2: CPU dengan utilisasi thread maksimal (seluruh core).
    
    Mengembalikan (device_str, info_dict).
    """
    cuda_available = torch.cuda.is_available()

    if preferred in ["auto", "cuda", "gpu"] and cuda_available:
        try:
            # Aktifkan cuDNN benchmark dan TF32 untuk Tensor Cores NVIDIA RTX
            torch.backends.cudnn.benchmark = True
            if hasattr(torch.backends.cuda, "matmul"):
                torch.backends.cuda.matmul.allow_tf32 = True
            if hasattr(torch.backends.cudnn, "allow_tf32"):
                torch.backends.cudnn.allow_tf32 = True

            dev_idx = 0
            dev_name = torch.cuda.get_device_name(dev_idx)
            total_vram_gb = torch.cuda.get_device_properties(dev_idx).total_memory / (1024 ** 3)
            logger.info(f"🚀 GPU NVIDIA Terdeteksi & Digunakan: {dev_name} ({total_vram_gb:.1f} GB VRAM)")
            
            info = {
                "type": "GPU",
                "device": "cuda:0",
                "name": dev_name,
                "vram_gb": round(total_vram_gb, 1),
                "desc": f"NVIDIA GPU: {dev_name} ({total_vram_gb:.1f} GB VRAM) - Mode Akselerasi Maksimal"
            }
            return "cuda:0", info
        except Exception as e:
            logger.warning(f"Gagal inisialisasi CUDA ({e}), beralih ke CPU.")

    # Fallback / Prioritas 2: CPU Maksimal
    cores = os.cpu_count() or 8
    try:
        torch.set_num_threads(cores)
        cv2.setNumThreads(cores)
        os.environ["OMP_NUM_THREADS"] = str(cores)
        os.environ["MKL_NUM_THREADS"] = str(cores)
    except Exception:
        pass

    logger.info(f"⚙️ Menggunakan CPU dengan {cores} threads paralel.")
    info = {
        "type": "CPU",
        "device": "cpu",
        "name": f"CPU ({cores} Cores)",
        "vram_gb": 0.0,
        "desc": f"CPU Multi-Threading ({cores} Cores Maksimal)"
    }
    return "cpu", info


def warmup_gpu_models(models: list, dummy_size: Tuple[int, int] = (640, 640)) -> None:
    """
    Melakukan 'warm-up' inference 1-kali pada GPU dengan dummy frame
    agar CUDA context dan kernel TensorRT/cuDNN sudah terkompilasi sebelum video diproses.
    """
    if not torch.cuda.is_available():
        return
    try:
        import numpy as np
        dummy_frame = np.zeros((dummy_size[1], dummy_size[0], 3), dtype=np.uint8)
        for m in models:
            if m is not None and hasattr(m, "predict"):
                try:
                    m.predict(dummy_frame)
                except Exception:
                    pass
        torch.cuda.synchronize()
        logger.info("🔥 GPU CUDA Warmup Selesai - Kernel siap eksekusi kecepatan penuh.")
    except Exception as e:
        logger.debug(f"GPU warmup skipped: {e}")


