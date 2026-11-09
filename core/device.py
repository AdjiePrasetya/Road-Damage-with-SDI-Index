"""
Modul Manajemen Komputasi Hardware Otomatis (GPU CUDA, Apple MPS & CPU Multi-threading).
Mendeteksi dan menggunakan GPU secara dinamis pada perangkat lokal manapun:
- Prioritas 1: NVIDIA GPU (CUDA) pada perangkat lokal pengguna (otomatis deteksi device & VRAM).
- Prioritas 2: Apple Silicon GPU (Metal / MPS) jika dijalankan pada Mac.
- Prioritas 3: CPU Multi-threading (100% utilitas seluruh core CPU) jika tanpa GPU.
Dilengkapi pendeteksi hardware fisik untuk mendiagnosis jika laptop memiliki GPU NVIDIA
namun PyTorch yang terpasang masih versi CPU.
"""
import os
import shutil
import subprocess
import logging
from typing import Tuple, Dict, Any, Optional
import torch
import cv2

logger = logging.getLogger(__name__)


def detect_physical_gpu() -> Optional[str]:
    """
    Mendeteksi keberadaan kartu grafis fisik (NVIDIA) di komputer pengguna,
    meskipun PyTorch yang aktif saat ini masih versi CPU.
    """
    # 1. Deteksi lewat nvidia-smi
    try:
        nvsmi = shutil.which("nvidia-smi")
        if not nvsmi and os.name == "nt":
            common_paths = [
                r"C:\Windows\System32\nvidia-smi.exe",
                r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe"
            ]
            for cp in common_paths:
                if os.path.exists(cp):
                    nvsmi = cp
                    break
        if nvsmi:
            res = subprocess.run(
                [nvsmi, "--query-gpu=name,memory.total", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=2
            )
            if res.returncode == 0 and res.stdout.strip():
                return res.stdout.strip().split("\n")[0].strip()
    except Exception:
        pass

    # 2. Deteksi lewat WMI (Windows) jika nvidia-smi tidak langsung merespons
    if os.name == "nt":
        try:
            res = subprocess.run(
                ["powershell", "-NoProfile", "-Command", "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name"],
                capture_output=True, text=True, timeout=3
            )
            if res.returncode == 0 and res.stdout.strip():
                lines = [line.strip() for line in res.stdout.strip().split("\n") if line.strip()]
                for l in lines:
                    if any(k in l.lower() for k in ["nvidia", "geforce", "quadro", "rtx", "gtx"]):
                        return l
        except Exception:
            pass

    # 3. Deteksi Linux driver path
    if os.path.exists("/proc/driver/nvidia/version"):
        try:
            with open("/proc/driver/nvidia/version", "r") as f:
                ver = f.readline().strip()
                return f"NVIDIA GPU ({ver})"
        except Exception:
            return "NVIDIA GPU"

    return None


def get_optimal_device(preferred: str = "auto") -> Tuple[str, Dict[str, Any]]:
    """
    Menentukan device komputasi terbaik secara otomatis dan dinamis di komputer manapun:
    - Prioritas 1: NVIDIA GPU (CUDA) jika terpasang dan PyTorch mendukung CUDA.
      Mengaktifkan cuDNN benchmark dan optimasi Tensor Cores TF32 secara dinamis.
    - Prioritas 2: Apple Silicon GPU (MPS) jika dijalankan pada Mac M1/M2/M3/M4.
    - Prioritas 3: CPU Multi-threading maksimal (seluruh core fisik/logis) jika tanpa GPU.
    
    Mengembalikan: (device_str, info_dict)
    """
    cuda_available = torch.cuda.is_available()

    # PRIORITAS 1: GPU NVIDIA (CUDA)
    if preferred in ["auto", "cuda", "gpu"] and cuda_available:
        try:
            dev_idx = torch.cuda.current_device()
            dev_name = torch.cuda.get_device_name(dev_idx)
            total_vram_gb = torch.cuda.get_device_properties(dev_idx).total_memory / (1024 ** 3)
            dev_str = f"cuda:{dev_idx}"

            # Aktifkan optimasi GPU Tensor Cores & cuDNN benchmark
            torch.backends.cudnn.benchmark = True
            if hasattr(torch.backends.cuda, "matmul"):
                torch.backends.cuda.matmul.allow_tf32 = True
            if hasattr(torch.backends.cudnn, "allow_tf32"):
                torch.backends.cudnn.allow_tf32 = True

            cuda_ver = torch.version.cuda or "Tersedia"
            logger.info(f"🚀 GPU NVIDIA Terdeteksi & Digunakan: {dev_name} ({total_vram_gb:.1f} GB VRAM) - CUDA {cuda_ver}")

            info = {
                "type": "GPU",
                "backend": "CUDA",
                "device": dev_str,
                "name": dev_name,
                "vram_gb": round(total_vram_gb, 1),
                "cuda_version": cuda_ver,
                "desc": f"NVIDIA GPU: {dev_name} ({total_vram_gb:.1f} GB VRAM) - Mode Akselerasi Maksimal",
                "torch_cuda_missing": False,
                "hardware_gpu_name": None
            }
            return dev_str, info
        except Exception as e:
            logger.warning(f"Gagal inisialisasi CUDA ({e}), mencoba fallback.")

    # PRIORITAS 2: Apple Silicon GPU (MPS)
    if preferred in ["auto", "mps", "gpu"] and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        try:
            logger.info("🍏 Apple Silicon GPU (Metal/MPS) Terdeteksi & Digunakan.")
            info = {
                "type": "GPU",
                "backend": "MPS",
                "device": "mps",
                "name": "Apple Silicon GPU (Metal)",
                "vram_gb": 0.0,
                "cuda_version": None,
                "desc": "Apple Silicon GPU (Metal Performance Shaders - MPS)",
                "torch_cuda_missing": False,
                "hardware_gpu_name": None
            }
            return "mps", info
        except Exception as e:
            logger.warning(f"Gagal inisialisasi Apple MPS ({e}), beralih ke CPU.")

    # DIAGNOSIS: Cek apakah laptop memiliki hardware GPU fisik tapi PyTorch masih versi CPU
    physical_gpu = detect_physical_gpu()
    torch_cuda_missing = physical_gpu is not None

    # PRIORITAS 3: CPU Multi-threading Maksimal
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
        "backend": "CPU",
        "device": "cpu",
        "name": f"CPU ({cores} Cores)",
        "vram_gb": 0.0,
        "cuda_version": None,
        "desc": f"CPU Multi-Threading ({cores} Cores Maksimal)",
        "torch_cuda_missing": torch_cuda_missing,
        "hardware_gpu_name": physical_gpu,
        "install_cuda_cmd": "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --upgrade"
    }
    return "cpu", info


def warmup_gpu_models(models: list, dummy_size: Tuple[int, int] = (640, 640)) -> None:
    """
    Melakukan 'warm-up' inference 1-kali pada GPU dengan dummy frame
    agar CUDA/MPS context dan kernel TensorRT/cuDNN sudah terkompilasi sebelum video diproses.
    """
    if not torch.cuda.is_available() and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
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
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        logger.info("🔥 GPU Warmup Selesai - Kernel siap eksekusi kecepatan penuh.")
    except Exception as e:
        logger.debug(f"GPU warmup skipped: {e}")
