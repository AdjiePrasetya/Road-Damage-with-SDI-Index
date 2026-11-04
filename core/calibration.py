"""
Modul Kalibrasi Otomatis (Auto-Calibration) Computer Vision untuk Survei Jalan.
Menghitung rasio meter-per-pixel, faktor luas retak, dan parameter sampling
secara otomatis berdasarkan dimensi frame video dan lebar fisik jalan.
"""
from typing import Tuple, Dict, Any
import numpy as np


def auto_calibrate_camera(
    frame_shape: Tuple[int, int, ...],
    road_width_m: float = 6.0
) -> Dict[str, Any]:
    """
    Melakukan kalibrasi otomatis parameter visual tanpa input manual pengguna.
    
    Pendekatan:
      - Kamera survei menghadap ke depan jalan dengan Field of View (FOV) standar ~70-90 derajat.
      - Pada sepertiga bawah frame (bidang jalan terdekat/ROI kontak ban),
        lebar jalan fisik mencakup sekitar 75% dari total lebar horizontal frame (W).
      - Maka: meter_per_pixel = road_width_m / (W * 0.75).
    """
    h = frame_shape[0]
    w = frame_shape[1]

    # Asumsi bidang jalan di sepertiga bawah mencakup 75% lebar piksel kamera
    road_pixel_width = max(100.0, float(w) * 0.75)
    meter_per_pixel = road_width_m / road_pixel_width

    # Faktor luas retak adaptif per tipe kerusakan (bukan hardcode satu angka untuk semua)
    # Retak buaya (alligator) mengisi area lebih padat dibanding retak garis tipis tunggal
    crack_area_factors = {
        "alligator crack": 0.35,      # Pola jala/kulit buaya mengisi ~35% bbox
        "longitudinal crack": 0.15,   # Garis retak vertikal tipis ~15% bbox
        "transverse crack": 0.15,     # Garis retak horizontal tipis ~15% bbox
        "other corruption": 0.25,     # Kerusakan acak ~25% bbox
        "Pothole": 0.785,             # Lubang berbentuk lingkaran/elips (pi/4 * bbox)
    }

    return {
        "meter_per_pixel": meter_per_pixel,
        "road_pixel_width": road_pixel_width,
        "crack_area_factors": crack_area_factors,
        "frame_width": w,
        "frame_height": h,
        "summary": f"Resolusi {w}x{h} px | Lebar jalan {road_width_m:.1f}m -> Kalibrasi: {meter_per_pixel*1000:.2f} mm/pixel"
    }


def auto_calculate_frame_step(fps: float) -> int:
    """
    Menghitung interval sampling frame optimal otomatis:
    Mengambil sampel kira-kira tiap 0.5 - 1.0 detik video
    agar video diproses cepat, stabil di RAM cloud, dan tetap mencakup seluruh jalan.
    """
    fps_val = max(1.0, fps)
    # Sampling tiap ~0.66 detik video (misal 30 fps -> tiap 20 frame)
    step = int(round(fps_val * 0.66))
    return max(5, min(step, 60))

