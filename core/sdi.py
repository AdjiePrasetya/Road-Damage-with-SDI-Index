"""
Modul Perhitungan Surface Distress Index (SDI)
Berdasarkan Manual Aplikasi Sistem Program Pemeliharaan Jalan (PKRMS)
SE Dirjen Bina Marga No. 22/SE/Db/2021 (Manual No. 04/M/BM/2021 Bab 4.2.5, Tabel 9 & Tabel 10).
"""
from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class SDIResult:
    """Hasil perhitungan SDI bertingkat sesuai Tabel 9 & Tabel 10."""
    sdi1: float                 # Nilai SDI Level 1 (% luas retak)
    sdi2: float                 # Nilai SDI Level 2 (lebar retak mm)
    sdi3: float                 # Nilai SDI Level 3 (jumlah lubang/km)
    sdi4: float                 # Nilai SDI Level 4 (alur bekas roda cm)
    final_sdi: float            # SDI Akhir = SDI4
    condition: str              # Baik, Sedang, Rusak Ringan, Rusak Berat
    color: str                  # Warna representasi kondisi (#hex)
    crack_area_pct: float       # Input % luas retak
    avg_crack_width_mm: float   # Input lebar retak (mm)
    potholes_per_km: float      # Input lubang/km
    rut_depth_cm: float         # Input alur (cm)


def compute_sdi_level_1(crack_area_pct: float) -> float:
    """
    Menghitung SDI Level 1 berdasarkan % luas retak (Tabel 9).
    Rumus:
      - None (0%)      -> SDI1 = 0
      - < 10% (0 < x < 10) -> SDI1 = 5
      - 10 - 30%       -> SDI1 = 20
      - > 30%          -> SDI1 = 40
    """
    if crack_area_pct <= 0.0:
        return 0.0
    elif crack_area_pct < 10.0:
        return 5.0
    elif crack_area_pct <= 30.0:
        return 20.0
    else:
        return 40.0


def compute_sdi_level_2(
    sdi1: float,
    avg_crack_width_mm: float,
    crack_width_threshold_mm: float = 3.0,
    multiplier: float = 2.0
) -> float:
    """
    Menghitung SDI Level 2 berdasarkan rata-rata lebar retak (Tabel 9).
    Rumus:
      - None / < 1 mm / 1 - 3 mm -> SDI2 = SDI1
      - > threshold_mm           -> SDI2 = SDI1 * multiplier (default 2.0)
    
    Catatan:
      Tabel 9 menetapkan threshold 3 mm, sedangkan narasi manual pada halaman 38
      menyebutkan 5 mm. Parameter crack_width_threshold_mm dibuat configurable.
    """
    if avg_crack_width_mm > crack_width_threshold_mm:
        return sdi1 * multiplier
    return sdi1


def compute_sdi_level_3(sdi2: float, potholes_per_km: float) -> float:
    """
    Menghitung SDI Level 3 berdasarkan jumlah lubang per km (Tabel 9).
    Rumus:
      - None (0)          -> SDI3 = SDI2
      - < 10 per km       -> SDI3 = SDI2 + 15
      - 10 - 50 per km    -> SDI3 = SDI2 + 75
      - > 50 per km       -> SDI3 = SDI2 + 225
    """
    if potholes_per_km <= 0.0:
        return sdi2
    elif potholes_per_km < 10.0:
        return sdi2 + 15.0
    elif potholes_per_km <= 50.0:
        return sdi2 + 75.0
    else:
        return sdi2 + 225.0


def compute_sdi_level_4(sdi3: float, rut_depth_cm: float) -> float:
    """
    Menghitung SDI Level 4 berdasarkan rata-rata kedalaman alur bekas roda (cm) (Tabel 9).
    Rumus:
      - None (0 cm)       -> SDI4 = SDI3
      - < 1 cm            -> SDI4 = SDI3 + 2.5
      - 1 - 3 cm          -> SDI4 = SDI3 + 10.0
      - > 3 cm            -> SDI4 = SDI3 + 20.0
    """
    if rut_depth_cm <= 0.0:
        return sdi3
    elif rut_depth_cm < 1.0:
        return sdi3 + 2.5
    elif rut_depth_cm <= 3.0:
        return sdi3 + 10.0
    else:
        return sdi3 + 20.0


def classify_sdi_condition(sdi: float) -> Dict[str, str]:
    """
    Mengklasifikasikan nilai SDI akhir ke dalam kategori kondisi jalan (Tabel 10).
    Rentang:
      - SDI < 50         : Baik (Good)         - Hijau
      - 50 <= SDI < 100  : Sedang (Fair)       - Kuning
      - 100 <= SDI < 150 : Rusak Ringan (Poor) - Oranye
      - SDI >= 150       : Rusak Berat (Bad)   - Merah
    """
    if sdi < 50.0:
        return {"label": "Baik", "color": "#2ECC71", "badge": "success"}
    elif sdi < 100.0:
        return {"label": "Sedang", "color": "#F1C40F", "badge": "warning"}
    elif sdi < 150.0:
        return {"label": "Rusak Ringan", "color": "#E67E22", "badge": "danger"}
    else:
        return {"label": "Rusak Berat", "color": "#E74C3C", "badge": "dark"}


def calculate_sdi(
    crack_area_pct: float,
    avg_crack_width_mm: float = 0.0,
    potholes_per_km: float = 0.0,
    rut_depth_cm: float = 0.0,
    crack_width_threshold_mm: float = 3.0,
    config: Optional[Dict[str, Any]] = None
) -> SDIResult:
    """
    Menghitung SDI secara lengkap dari Level 1 sampai Level 4 sesuai Manual PKRMS.
    """
    threshold_w = crack_width_threshold_mm
    multiplier_w = 2.0

    if config and "sdi_table_9" in config:
        cfg9 = config["sdi_table_9"]
        if "crack_width" in cfg9:
            threshold_w = cfg9["crack_width"].get("threshold_mm", threshold_w)
            multiplier_w = cfg9["crack_width"].get("multiplier", multiplier_w)

    sdi1 = compute_sdi_level_1(crack_area_pct)
    sdi2 = compute_sdi_level_2(sdi1, avg_crack_width_mm, threshold_w, multiplier_w)
    sdi3 = compute_sdi_level_3(sdi2, potholes_per_km)
    sdi4 = compute_sdi_level_4(sdi3, rut_depth_cm)

    final_sdi = float(sdi4)
    cond_info = classify_sdi_condition(final_sdi)

    return SDIResult(
        sdi1=sdi1,
        sdi2=sdi2,
        sdi3=sdi3,
        sdi4=sdi4,
        final_sdi=final_sdi,
        condition=cond_info["label"],
        color=cond_info["color"],
        crack_area_pct=round(crack_area_pct, 2),
        avg_crack_width_mm=round(avg_crack_width_mm, 2),
        potholes_per_km=round(potholes_per_km, 2),
        rut_depth_cm=round(rut_depth_cm, 2)
    )

