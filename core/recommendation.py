"""
Modul Saran Penanganan dan Rekomendasi Teknis Jalan
Berdasarkan Manual PKRMS (Adaptasi Tabel 3 & Standar Ditjen Bina Marga).
"""
from typing import Dict, Any, Optional, Tuple


DEFAULT_ASPHALT_RECOMMENDATIONS = {
    "Baik": "Pemeliharaan rutin (pembersihan drainase, tambal kecil, pemeliharaan bahu)",
    "Sedang": "Pemeliharaan rutin intensif / pemeliharaan berkala ringan (tambal retak, crack sealing, fog seal / laburan aspal)",
    "Rusak Ringan": "Pemeliharaan berkala (patching lubang + overlay / lapis ulang tipis)",
    "Rusak Berat": "Rehabilitasi / peningkatan struktur (overlay tebal, perbaikan struktur, rekonstruksi setempat)",
}

DEFAULT_CONCRETE_RECOMMENDATIONS = {
    "Baik": "Pemeliharaan rutin (pembersihan sambungan / joint & saluran samping)",
    "Sedang": "Perbaikan sambungan pelat (joint resealing) & penutupan retak rambut (crack sealing)",
    "Rusak Ringan": "Penambalan kedalaman parsial (partial depth repair), penjahitan retak (cross stitching), & injeksi semen/grouting",
    "Rusak Berat": "Penggantian pelat beton penuh (full-depth slab replacement) & stabilisasi tanah dasar",
}

DEFAULT_UNPAVED_RECOMMENDATIONS = {
    "Baik": "Perataan permukaan ringan & pembersihan saluran tepi",
    "Sedang": "Perataan dengan grader (grading) & pemadatan ulang setempat",
    "Rusak Ringan": "Penambahan material agregat pilihan (regravelling) & perataan jalan",
    "Rusak Berat": "Peningkatan struktur jalan (pembentukan kembali lapis pondasi / peningkatan ke aspal/beton)",
}


def get_recommendation(
    condition: str,
    pavement_type: str,
    config: Optional[Dict[str, Any]] = None
) -> Tuple[str, str]:
    """
    Memberikan rekomendasi perbaikan jalan berdasarkan kondisi SDI dan tipe perkerasan.
    Mengembalikan:
      (rekomendasi_teks, catatan_tambahan)
    """
    cfg_rec = config.get("recommendations", {}) if config else {}
    pave_lower = (pavement_type or "").lower()

    catatan = ""
    if "beton" in pave_lower or "rigid" in pave_lower:
        catatan = "Catatan: SDI manual hanya baku untuk jalan beraspal; saran bersifat indikatif teknis."
        beton_map = cfg_rec.get("beton", DEFAULT_CONCRETE_RECOMMENDATIONS)
        rec = beton_map.get(condition, DEFAULT_CONCRETE_RECOMMENDATIONS.get(condition, "Evaluasi teknis lapangan"))
    elif any(k in pave_lower for k in ["kerikil", "tanah", "unpaved"]):
        catatan = "Catatan: SDI dirancang untuk perkerasan beraspal; evaluasi perkerasan tanpa penutup."
        unpaved_map = cfg_rec.get("unpaved", DEFAULT_UNPAVED_RECOMMENDATIONS)
        rec = unpaved_map.get(condition, DEFAULT_UNPAVED_RECOMMENDATIONS.get(condition, "Evaluasi teknis lapangan"))
    else:
        # Default aspal (Hotmix, Lapen, Aspal)
        aspal_map = cfg_rec.get("aspal", DEFAULT_ASPHALT_RECOMMENDATIONS)
        rec = aspal_map.get(condition, DEFAULT_ASPHALT_RECOMMENDATIONS.get(condition, "Pemeliharaan sesuai standar teknis"))

    return rec, catatan

