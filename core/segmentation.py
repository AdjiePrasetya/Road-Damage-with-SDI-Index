"""
Modul Segmentasi Ruas Jalan Sesuai Logika PKRMS.
Membagi ruas menjadi blok dasar 100 m, memotong segmen jika jenis perkerasan berubah,
menggabungkan segmen sangat pendek (<10m), dan menyediakan dukungan koreksi manual.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from collections import Counter
import copy


@dataclass
class FrameRecord:
    """Informasi hasil inferensi pada satu frame."""
    frame_idx: int
    sta_m: float
    pavement_type: str
    pavement_conf: float
    crack_area_m2: float = 0.0
    crack_width_mm: float = 0.0
    pothole_ids: List[int] = field(default_factory=list)
    lat: Optional[float] = None
    lon: Optional[float] = None


@dataclass
class RoadSegment:
    """Representasi satu segmen jalan hasil pembagian & perhitungan."""
    road_name: str
    sta_start: float
    sta_end: float
    length_m: float
    pavement_type: str
    sdi_value: float = 0.0
    condition: str = "Baik"
    recommendation: str = ""
    crack_area_pct: float = 0.0
    avg_crack_width_mm: float = 0.0
    potholes_per_km: float = 0.0
    rut_depth_cm: float = 0.0
    flag: str = "Normal"
    color: str = "#2ECC71"
    frame_indices: List[int] = field(default_factory=list)
    unique_potholes: int = 0
    total_crack_area_m2: float = 0.0
    is_manual_override: bool = False

    @property
    def sta_start_m(self) -> float:
        return self.sta_start

    @sta_start_m.setter
    def sta_start_m(self, val: float) -> None:
        self.sta_start = float(val)

    @property
    def sta_end_m(self) -> float:
        return self.sta_end

    @sta_end_m.setter
    def sta_end_m(self, val: float) -> None:
        self.sta_end = float(val)

    @property
    def sta_start_str(self) -> str:
        km = int(self.sta_start // 1000)
        m = int(round(self.sta_start % 1000))
        return f"STA {km}+{m:03d}"

    @property
    def sta_end_str(self) -> str:
        km = int(self.sta_end // 1000)
        m = int(round(self.sta_end % 1000))
        return f"STA {km}+{m:03d}"


def smooth_pavement_types(
    frames: List[FrameRecord],
    window_size: int = 5,
    min_run_length: int = 3
) -> List[str]:
    """
    Melakukan smoothing pada sekuens klasifikasi jenis perkerasan.
    Menggunakan moving majority vote dilanjutkan dengan filter run-length
    agar tidak terjadi flicker/noise antar frame.
    """
    if not frames:
        return []

    raw_labels = [f.pavement_type for f in frames]
    n = len(raw_labels)
    if n <= window_size:
        # Gunakan mayoritas keseluruhan jika data sedikit
        majority = Counter(raw_labels).most_common(1)[0][0]
        return [majority] * n

    # Step 1: Moving Majority Vote
    half_w = window_size // 2
    step1_labels = []
    for i in range(n):
        start = max(0, i - half_w)
        end = min(n, i + half_w + 1)
        sub = raw_labels[start:end]
        maj = Counter(sub).most_common(1)[0][0]
        step1_labels.append(maj)

    # Step 2: Minimum Run-Length Filter
    # Jika sebuah run lebih pendek dari min_run_length, serap ke tetangga sebelumnya
    smoothed = list(step1_labels)
    current_run_start = 0
    for i in range(1, n + 1):
        if i == n or smoothed[i] != smoothed[current_run_start]:
            run_len = i - current_run_start
            if run_len < min_run_length:
                # Ganti dengan label sebelumnya jika ada, atau sesudahnya
                replacement = smoothed[current_run_start - 1] if current_run_start > 0 else (
                    smoothed[i] if i < n else smoothed[current_run_start]
                )
                for k in range(current_run_start, i):
                    smoothed[k] = replacement
            current_run_start = i

    return smoothed


def create_segments_from_frames(
    frames: List[FrameRecord],
    road_name: str = "Ruas Jalan Survei",
    sta_start_m: Optional[float] = None,
    base_segment_length_m: float = 100.0,
    min_segment_length_m: float = 10.0,
    smoothing_window: int = 5,
    min_run_length: int = 3
) -> List[RoadSegment]:
    """
    Membagi ruas menjadi segmen-segmen jalan sesuai aturan PKRMS:
    1. Blok dasar tiap base_segment_length_m (default 100m), dimulai dari STA awal jalan terdeteksi.
    2. Jika jenis perkerasan di dalam blok 100m berubah, blok dipotong tepat di STA perubahan.
    3. Segmen sangat pendek (< min_segment_length_m) digabung ke segmen tetangga & diberi flag.
    """
    if not frames:
        return []

    # Pastikan frame terurut berdasarkan STA
    sorted_frames = sorted(frames, key=lambda f: f.sta_m)
    smoothed_labels = smooth_pavement_types(
        sorted_frames,
        window_size=smoothing_window,
        min_run_length=min_run_length
    )

    min_sta = sorted_frames[0].sta_m
    max_sta = sorted_frames[-1].sta_m

    # Gunakan min_sta hasil deteksi model jika sta_start_m tidak ditentukan secara eksplisit
    overall_start = min_sta if sta_start_m is None else min(sta_start_m, min_sta)
    overall_end = max_sta
    if overall_end <= overall_start:
        overall_end = round(overall_start + max(1.0, float(len(sorted_frames))), 1)

    # Tentukan batas-batas blok dasar 100m
    block_start = overall_start
    raw_segments: List[RoadSegment] = []

    while block_start < overall_end:
        block_end = block_start + base_segment_length_m
        if block_end > overall_end:
            block_end = overall_end

        # Cari frame yang berada di dalam blok [block_start, block_end]
        block_frame_indices = [
            i for i, f in enumerate(sorted_frames)
            if block_start <= f.sta_m <= block_end
        ]

        if not block_frame_indices:
            # Tidak ada frame di rentang ini (misal jeda data)
            block_start = block_end
            continue

        # Periksa apakah jenis perkerasan berubah di dalam blok ini
        current_sub_start = block_start
        current_pavement = smoothed_labels[block_frame_indices[0]]
        current_sub_frames = []

        for idx in block_frame_indices:
            f = sorted_frames[idx]
            label = smoothed_labels[idx]

            if label != current_pavement:
                # Titik potong STA perubahan perkerasan
                cut_sta = round(f.sta_m, 1)
                sub_len = cut_sta - current_sub_start

                if sub_len > 0:
                    raw_segments.append(RoadSegment(
                        road_name=road_name,
                        sta_start=round(current_sub_start, 1),
                        sta_end=cut_sta,
                        length_m=round(sub_len, 1),
                        pavement_type=current_pavement,
                        flag="Dipotong (Perubahan Perkerasan)" if current_sub_start > block_start or cut_sta < block_end else "Normal",
                        frame_indices=current_sub_frames
                    ))

                # Mulai subsegmen baru setelah titik potong
                current_sub_start = cut_sta
                current_pavement = label
                current_sub_frames = [idx]
            else:
                current_sub_frames.append(idx)

        # Tambahkan sisa subsegmen hingga akhir blok
        sub_len = block_end - current_sub_start
        if sub_len > 0:
            raw_segments.append(RoadSegment(
                road_name=road_name,
                sta_start=round(current_sub_start, 1),
                sta_end=round(block_end, 1),
                length_m=round(sub_len, 1),
                pavement_type=current_pavement,
                flag="Dipotong (Perubahan Perkerasan)" if current_sub_start > block_start else "Normal",
                frame_indices=current_sub_frames
            ))

        block_start = block_end

    # Step 3: Penggabungan segmen sangat pendek (< min_segment_length_m)
    final_segments: List[RoadSegment] = []
    i = 0
    while i < len(raw_segments):
        seg = raw_segments[i]
        if seg.length_m < min_segment_length_m and len(raw_segments) > 1:
            # Gabungkan ke segmen tetangga sebelumnya jika ada, jika tidak ke sesudahnya
            if final_segments:
                prev_seg = final_segments[-1]
                prev_seg.sta_end = seg.sta_end
                prev_seg.length_m = round(prev_seg.sta_end - prev_seg.sta_start, 1)
                prev_seg.frame_indices.extend(seg.frame_indices)
                prev_seg.flag = f"Gabungan segmen pendek (<{min_segment_length_m:.0f}m)"
            elif i + 1 < len(raw_segments):
                next_seg = raw_segments[i + 1]
                next_seg.sta_start = seg.sta_start
                next_seg.length_m = round(next_seg.sta_end - next_seg.sta_start, 1)
                next_seg.frame_indices = seg.frame_indices + next_seg.frame_indices
                next_seg.flag = f"Gabungan segmen pendek (<{min_segment_length_m:.0f}m)"
            else:
                final_segments.append(seg)
        else:
            final_segments.append(seg)
        i += 1

    return final_segments


def apply_manual_corrections(
    segments: List[RoadSegment],
    user_overrides: List[Dict[str, Any]]
) -> List[RoadSegment]:
    """
    Menerapkan koreksi manual dari pengguna pada segmen yang ada.
    user_overrides berisi daftar dict:
      [{'index': 0, 'sta_start': 0, 'sta_end': 50, 'pavement_type': 'Hotmix', 'rut_depth_cm': 1.5}, ...]
    """
    updated_segments = copy.deepcopy(segments)
    for ov in user_overrides:
        idx = ov.get("index")
        if idx is not None and 0 <= idx < len(updated_segments):
            seg = updated_segments[idx]
            if "sta_start" in ov:
                seg.sta_start = float(ov["sta_start"])
            if "sta_end" in ov:
                seg.sta_end = float(ov["sta_end"])
            seg.length_m = round(seg.sta_end - seg.sta_start, 1)
            if "pavement_type" in ov:
                seg.pavement_type = str(ov["pavement_type"])
            if "rut_depth_cm" in ov:
                seg.rut_depth_cm = float(ov["rut_depth_cm"])
            seg.flag = "Koreksi Manual"
            seg.is_manual_override = True

    return updated_segments

