"""
Dashboard Survei Kondisi Jalan Berbasis SDI (Surface Distress Index)
Sesuai Manual PKRMS (SE Dirjen Bina Marga No. 22/SE/Db/2021, Manual No. 04/M/BM/2021 Bab 4.2.5).

Direktorat Jenderal Bina Marga - Kementerian Pekerjaan Umum dan Perumahan Rakyat.
Dioptimalkan untuk Deployment Streamlit Cloud (Bebas Chunk Error & Cache Mismatch).
"""
import os
import yaml
import tempfile
import subprocess
import logging
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk

# Import modul core dan model adapter
from core.device import get_optimal_device
from core.sdi import calculate_sdi, classify_sdi_condition
from core.segmentation import RoadSegment, apply_manual_corrections
from core.recommendation import get_recommendation
from core.video import process_video_survey
from core.exporter import export_to_csv, export_to_excel, export_to_html_report, segments_to_dataframe
from core.demo_data import generate_demo_survey_data
from models.damage_detector import RoadDamageDetector
from models.pavement_classifier import PavementClassifier
from models.sta_estimator import StaEstimator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DashboardSDI")

# ==============================================================================
# KONFIGURASI HALAMAN STREAMLIT
# ==============================================================================
st.set_page_config(
    page_title="Dashboard Survei SDI - PKRMS Bina Marga",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling CSS
st.markdown("""
<style>
    .main-header {
        font-size: 26px;
        font-weight: 700;
        color: #1F497D;
        margin-bottom: 2px;
    }
    .sub-header {
        font-size: 14px;
        color: #555;
        margin-bottom: 20px;
    }
    .kpi-card {
        background-color: #F8F9FA;
        border-radius: 8px;
        padding: 16px;
        border-left: 5px solid #1E88E5;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }
    .kpi-title {
        font-size: 13px;
        color: #666;
        font-weight: 600;
        text-transform: uppercase;
    }
    .kpi-val {
        font-size: 24px;
        font-weight: bold;
        color: #1F497D;
        margin-top: 4px;
    }
    .kpi-note {
        font-size: 12px;
        color: #888;
        margin-top: 2px;
    }
    .auto-badge {
        background-color: #E8F4FD;
        border: 1px solid #BEE3F8;
        border-radius: 6px;
        padding: 10px 14px;
        font-size: 12px;
        color: #2B6CB0;
        margin-bottom: 15px;
    }
    .video-card {
        background-color: #F8F9FA;
        border-radius: 8px;
        padding: 16px;
        border: 1px solid #E2E8F0;
        margin-bottom: 20px;
    }
    /* Tabel HTML Khusus SDI */
    .sdi-table-wrapper {
        width: 100%;
        overflow-x: auto;
        margin-top: 15px;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
    }
    .sdi-table {
        width: 100%;
        border-collapse: collapse;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        font-size: 13px;
    }
    .sdi-table th {
        background-color: #1F497D;
        color: #FFFFFF;
        padding: 12px 14px;
        text-align: center;
        font-weight: 600;
        white-space: nowrap;
    }
    .sdi-table td {
        padding: 10px 14px;
        border-bottom: 1px solid #EDF2F7;
        color: #2D3748;
    }
    .sdi-table tr:hover {
        background-color: #F7FAFC;
    }
    .badge-cond {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 12px;
        text-align: center;
    }
    .badge-baik { background-color: #D4EDDA; color: #155724; border: 1px solid #C3E6CB; }
    .badge-sedang { background-color: #FFF3CD; color: #856404; border: 1px solid #FFEEBA; }
    .badge-ringan { background-color: #FFE5D0; color: #B35300; border: 1px solid #FFD1B3; }
    .badge-berat { background-color: #F8D7DA; color: #721C24; border: 1px solid #F5C6CB; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { padding: 10px 18px; font-weight: 600; border-radius: 4px 4px 0 0; }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# CACHING & INITIALIZATION
# ==============================================================================
@st.cache_data
def load_config():
    cfg_path = "config.yaml"
    if os.path.exists(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}

CONFIG = load_config()


@st.cache_resource
def load_models():
    logger.info("Memuat model deep learning & adapter...")
    damage_cfg = CONFIG.get("models", {}).get("road_damage", {})
    pave_cfg = CONFIG.get("models", {}).get("perkerasan_clf", {})
    gps_cfg = CONFIG.get("models", {}).get("gps_sta", {})

    damage_det = RoadDamageDetector(
        cracks_model_path=damage_cfg.get("cracks_path", "road_damage_models/cracks/weights/best.pt"),
        potholes_model_path=damage_cfg.get("potholes_path", "road_damage_models/potholes/weights/best.pt"),
        rutting_model_path=damage_cfg.get("rutting_path", "road_damage_models/rutting/weights/best.pt"),
        conf_threshold=damage_cfg.get("confidence_threshold", 0.25),
        iou_threshold=damage_cfg.get("iou_threshold", 0.45),
        device=damage_cfg.get("device", "auto"),
        class_mapping=CONFIG.get("class_mapping", {}).get("road_damage", None)
    )
    pave_clf = PavementClassifier(
        model_path=pave_cfg.get("primary_path", "perkerasan_clf_models/runs/perkerasan_cls/weights/best.pt"),
        device=pave_cfg.get("device", "auto"),
        class_mapping=CONFIG.get("class_mapping", {}).get("perkerasan", None)
    )
    sta_est = StaEstimator(
        model_path=gps_cfg.get("model_path", "gps_sta_model/weights/yolo26n.pt"),
        default_speed_kmh=gps_cfg.get("default_speed_kmh", 30.0),
        device=gps_cfg.get("device", "auto")
    )
    return damage_det, pave_clf, sta_est


# Inisialisasi session state
if "segments" not in st.session_state:
    st.session_state.segments = []
if "preview_frames" not in st.session_state:
    st.session_state.preview_frames = []
if "survey_stats" not in st.session_state:
    st.session_state.survey_stats = {}
if "annotated_video_path" not in st.session_state:
    st.session_state.annotated_video_path = None
if "detected_road_name" not in st.session_state:
    st.session_state.detected_road_name = ""
if "road_info" not in st.session_state:
    st.session_state.road_info = {
        "road_name": "",
        "road_width_m": 6.0,
        "base_segment_len": 100.0,
    }


# ==============================================================================
# HEADER APLIKASI
# ==============================================================================
st.markdown('<div class="main-header">🛣️ Dashboard Survei Kondisi Jalan - Metode SDI (PKRMS)</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Berdasarkan <strong>Manual PKRMS</strong> (SE Dirjen Bina Marga No. 22/SE/Db/2021, '
    'Manual No. 04/M/BM/2021 Bab 4.2.5, Tabel 9 & 10) • Khusus Surface Distress Index (SDI)</div>',
    unsafe_allow_html=True
)

# Cek & Deteksi Otomatis Hardware Komputasi (GPU / CPU)
_, initial_dev_info = get_optimal_device()
if initial_dev_info.get("torch_cuda_missing"):
    hw_gpu_name = initial_dev_info.get("hardware_gpu_name", "NVIDIA GPU")
    st.warning(
        f"💡 **Hardware GPU Terdeteksi di Komputer Anda: `{hw_gpu_name}`**, "
        "tetapi pustaka PyTorch yang aktif di lingkungan ini saat ini adalah versi CPU.\n\n"
        "Agar dashboard otomatis menjalankan seluruh model AI menggunakan GPU Anda dengan akselerasi maksimal, "
        "silakan jalankan perintah berikut di Terminal / Command Prompt:\n\n"
        f"```bash\n{initial_dev_info.get('install_cuda_cmd')}\n```\n\n"
        "Setelah instalasi selesai, refresh halaman ini (`F5`) dan dashboard akan langsung beralih ke GPU secara otomatis!"
    )


# ==============================================================================
# SIDEBAR: PENGATURAN SURVEI RINGKAS & OTOMATIS
# ==============================================================================
with st.sidebar:
    st.header("⚙️ Pengaturan Survei")

    uploaded_video = st.file_uploader(
        "📹 Upload Video Survei (MP4 / MOV / AVI)",
        type=["mp4", "mov", "avi"],
        help="Bebas ukuran file (batas upload telah dinaikkan hingga 100 GB)"
    )

    local_video_path = st.text_input(
        "📁 Atau Masukkan Path Video Lokal (Opsional)",
        value="",
        placeholder=r"Misal: D:\Project\video_survei.mov",
        help="Jika ukuran file video sangat masif, Anda bisa langsung memasukkan path file lokal di komputer agar proses instan tanpa waktu tunggu upload browser."
    )

    # EKSTRAKSI OTOMATIS NAMA JALAN DARI FILE VIDEO
    active_video_name = None
    if uploaded_video is not None:
        active_video_name = uploaded_video.name
    elif local_video_path and os.path.exists(local_video_path):
        active_video_name = os.path.basename(local_video_path)

    clean_name = ""
    if active_video_name:
        raw_name = os.path.splitext(active_video_name)[0]
        clean_name = raw_name.replace("_", " ").replace("-", " ").strip().title()
        st.session_state.detected_road_name = clean_name

    with st.expander("📌 Identitas Ruas Jalan", expanded=True):
        road_name = st.text_input(
            "Nama Ruas Jalan",
            value=st.session_state.detected_road_name,
            placeholder="Belum ada video (otomatis terisi saat video dimasukkan)",
            help="Otomatis diambil dari nama file video yang diunggah, atau isi manual jika diperlukan"
        )
        road_width = st.number_input(
            "Lebar Jalan (m)",
            value=6.0, step=0.5, min_value=1.0,
            help="Lebar perkerasan fisik jalan untuk kalkulasi luas retak dan kalibrasi otomatis"
        )

    with st.expander("📏 Parameter Segmentasi PKRMS", expanded=True):
        base_segment_len = st.number_input(
            "Panjang Blok Dasar (m)",
            value=100.0, step=25.0, min_value=25.0,
            help="Panjang standar segmen evaluasi PKRMS (Tabel 9: tiap 100 m)"
        )

    # Info Sistem Otomatis
    st.markdown("""
    <div class="auto-badge">
        <strong>⚡ Otomasi Penuh (Model AI Deep Learning):</strong><br>
        • Bebas batas ukuran upload video (kuota hingga 100 GB).<br>
        • Opsi input path lokal langsung tanpa waktu upload.<br>
        • Nama ruas jalan otomatis dari nama file video.<br>
        • Model <code>cracks/</code>: deteksi luas & lebar retak otomatis.<br>
        • Model <code>potholes/</code>: deteksi & hitung lubang/km otomatis.<br>
        • Model <code>rutting/</code>: prediksi kedalaman alur otomatis.<br>
        • Model <code>gps_sta_model/</code>: hitung jarak & STA mandiri.<br>
        • Kalibrasi kamera & resolusi dihitung otomatis.
    </div>
    """, unsafe_allow_html=True)

    # Indikator Komputasi Hardware di Sidebar
    if initial_dev_info.get("type") == "GPU":
        st.markdown(f"""
        <div style="background-color: #E8F8F5; border: 1px solid #A3E4D7; border-radius: 6px; padding: 8px 12px; margin-bottom: 12px; font-size: 12px; color: #117A65;">
            🚀 <strong>GPU Aktif:</strong> {initial_dev_info.get('name')} ({initial_dev_info.get('vram_gb', 0)} GB VRAM)
        </div>
        """, unsafe_allow_html=True)
    elif initial_dev_info.get("torch_cuda_missing"):
        st.markdown(f"""
        <div style="background-color: #FEF9E7; border: 1px solid #F9E79F; border-radius: 6px; padding: 8px 12px; margin-bottom: 12px; font-size: 12px; color: #7D6608;">
            ⚠️ <strong>GPU Terdeteksi:</strong> {initial_dev_info.get('hardware_gpu_name')}<br><small>Perlu setup PyTorch CUDA agar aktif</small>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="background-color: #EBF5FB; border: 1px solid #AED6F1; border-radius: 6px; padding: 8px 12px; margin-bottom: 12px; font-size: 12px; color: #1B4F72;">
            ⚙️ <strong>Mode CPU:</strong> {initial_dev_info.get('name')}
        </div>
        """, unsafe_allow_html=True)

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        process_btn = st.button("🚀 Proses Video", type="primary", use_container_width=True)
    with col_btn2:
        demo_btn = st.button("🧪 Jalankan Demo", use_container_width=True)

# Simpan info ke state
st.session_state.road_info = {
    "road_name": road_name.strip(),
    "road_width_m": road_width,
    "base_segment_len": base_segment_len
}


# ==============================================================================
# LOGIKA AKSI: PROSES VIDEO & DEMO MODE
# ==============================================================================
if demo_btn:
    demo_road_name = road_name.strip() if road_name.strip() else "Jalan Ilyas Bedul (Demo)"
    st.session_state.detected_road_name = demo_road_name
    st.session_state.road_info["road_name"] = demo_road_name
    with st.spinner(f"Memuat data demo survei realistis untuk '{demo_road_name}' & menyiapkan video inferensi..."):
        demo_segs, demo_previews, demo_stats, demo_video_path = generate_demo_survey_data(
            road_name=demo_road_name,
            sta_start=0.0,
            road_width_m=road_width,
            config=CONFIG
        )
        st.session_state.segments = demo_segs
        st.session_state.preview_frames = demo_previews
        st.session_state.survey_stats = demo_stats
        st.session_state.annotated_video_path = demo_video_path
        st.success(f"✅ Mode Demo dimuat untuk {demo_road_name}! Video inferensi siap ditonton dan diunduh.")

if process_btn:
    video_source_path = None
    is_temp_file = False
    display_name = ""

    if uploaded_video is not None:
        display_name = uploaded_video.name
        suffix = os.path.splitext(uploaded_video.name)[1] or ".mp4"
        with st.spinner(f"Menyiapkan video '{uploaded_video.name}' ke ruang kerja (streaming hemat RAM)..."):
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            while True:
                chunk = uploaded_video.read(16 * 1024 * 1024)  # 16 MB per chunk
                if not chunk:
                    break
                tfile.write(chunk)
            tfile.close()
        video_source_path = tfile.name
        is_temp_file = True
    elif local_video_path and os.path.exists(local_video_path):
        video_source_path = local_video_path
        display_name = os.path.basename(local_video_path)
        is_temp_file = False
    else:
        st.error("⚠️ Silakan upload file video survei, masukkan path file video lokal yang valid, atau klik tombol 'Jalankan Demo'.")

    if video_source_path:
        exec_road_name = road_name.strip() if road_name.strip() else (clean_name if active_video_name else "Ruas Jalan")
        st.session_state.detected_road_name = exec_road_name
        st.session_state.road_info["road_name"] = exec_road_name

        damage_det, pave_clf, sta_est = load_models()

        progress_bar = st.progress(0.0)
        status_text = st.empty()

        def update_progress(ratio, msg):
            progress_bar.progress(min(1.0, max(0.0, ratio)))
            status_text.text(msg)

        try:
            with st.spinner(f"Menganalisis video '{display_name}' dengan 3 Model Segmentasi Kerusakan & GPS..."):
                segs, previews, stats, out_vid_path = process_video_survey(
                    video_path=video_source_path,
                    damage_detector=damage_det,
                    pavement_classifier=pave_clf,
                    sta_estimator=sta_est,
                    road_name=exec_road_name,
                    road_width_m=road_width,
                    base_segment_length_m=base_segment_len,
                    config=CONFIG,
                    progress_callback=update_progress
                )

                st.session_state.segments = segs
                st.session_state.preview_frames = previews
                st.session_state.survey_stats = stats
                st.session_state.annotated_video_path = out_vid_path
                st.success(f"🎉 Pemrosesan selesai untuk {exec_road_name}! Total panjang jalan terdeteksi: {stats.get('total_length_m', 0):.1f} meter ({len(segs)} segmen).")
        except Exception as e:
            st.error(f"❌ Terjadi kesalahan saat memproses video: {e}")
            logger.exception("Error process_video_survey")
        finally:
            if is_temp_file and video_source_path and os.path.exists(video_source_path):
                try:
                    os.remove(video_source_path)
                except Exception:
                    pass
            progress_bar.empty()
            status_text.empty()


# ==============================================================================
# TAB UTAMA DASHBOARD
# ==============================================================================
tab_proses, tab_hasil, tab_vis, tab_peta, tab_koreksi, tab_export = st.tabs([
    "📸 Video Inferensi & Preview",
    "📋 Hasil SDI (Utama)",
    "📊 Visualisasi & KPI",
    "🗺️ Peta Spasial",
    "✏️ Koreksi Manual",
    "📥 Ekspor Laporan"
])


# ------------------------------------------------------------------------------
# TAB 1: VIDEO INFERENSI & PREVIEW FRAME
# ------------------------------------------------------------------------------
with tab_proses:
    curr_road = st.session_state.road_info.get("road_name", "").strip()
    if curr_road:
        st.subheader(f"🎬 Video Hasil Inferensi Model - {curr_road}")
    else:
        st.subheader("🎬 Video Hasil Inferensi Model")

    video_path = st.session_state.annotated_video_path
    if video_path and os.path.exists(video_path):
        file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
        abs_video_path = os.path.abspath(video_path)

        col_v1, col_v2 = st.columns([3, 1])
        with col_v1:
            st.markdown('<div class="video-card">', unsafe_allow_html=True)
            # Tampilkan langsung di browser jika ukuran video wajar (<= 200MB)
            if file_size_mb <= 200:
                try:
                    st.video(video_path)
                except Exception as ex_vid:
                    st.warning(f"Browser tidak dapat membuffer video ke memori ({ex_vid}). Gunakan tombol pemutar media lokal di sebelah kanan.")
            else:
                st.info(
                    f"🎬 **Video Hasil Inferensi Siap Disimpan di Disk Lokal:**\n\n"
                    f"`{os.path.basename(abs_video_path)}` (**{file_size_mb:.1f} MB**)\n\n"
                    "Karena aplikasi berjalan secara **offline di perangkat lokal**, video berukuran besar tidak perlu "
                    "dipaksa dimuat ke memori browser yang dapat memicu error RAM (*MemoryError*). Silakan gunakan tombol di panel "
                    "kanan untuk memutar langsung menggunakan pemutar media lokal (VLC / Windows Media Player) secara instan!"
                )
            st.markdown('</div>', unsafe_allow_html=True)

        with col_v2:
            st.markdown("#### ⚡ Akses & Pemutar Video Lokal")
            st.caption(f"📁 **Path:** `{abs_video_path}`\n\n📦 **Ukuran File:** {file_size_mb:.1f} MB")

            # Tombol 1: Buka & Putar Langsung di Media Player Lokal
            if st.button("▶️ Putar di Media Player Lokal", type="primary", use_container_width=True):
                try:
                    os.startfile(abs_video_path)
                    st.toast("Membuka video di pemutar media lokal...", icon="🎬")
                except Exception as ex_open:
                    st.error(f"Gagal membuka pemutar media: {ex_open}")

            # Tombol 2: Buka Folder di Windows Explorer
            if st.button("📂 Buka Lokasi di File Explorer", use_container_width=True):
                try:
                    subprocess.Popen(f'explorer /select,"{abs_video_path}"')
                    st.toast("Membuka Windows File Explorer...", icon="📂")
                except Exception as ex_exp:
                    st.error(f"Gagal membuka explorer: {ex_exp}")

            st.markdown("---")
            # Tombol unduh browser hanya jika ukuran wajar (< 150MB) untuk mencegah MemoryError dari vf.read()
            if file_size_mb <= 150:
                try:
                    with open(video_path, "rb") as vf:
                        video_bytes = vf.read()
                    st.download_button(
                        label="📥 Salin / Unduh MP4 (.mp4)",
                        data=video_bytes,
                        file_name=f"inferensi_sdi_{st.session_state.road_info.get('road_name', 'jalan')}.mp4",
                        mime="video/mp4",
                        use_container_width=True
                    )
                except Exception:
                    st.success("✅ File tersimpan permanen di folder `output_survei/`.")
            else:
                st.success("✅ File tersimpan permanen di folder `output_survei/`.")

            st.metric("Total Segmen", f"{len(st.session_state.segments)} segmen")
            st.metric("Panjang Terdeteksi", f"{st.session_state.survey_stats.get('total_length_m', 0):.1f} m")
    else:
        st.info("💡 Belum ada video yang diproses. Silakan upload video survei Anda di sidebar atau klik **'🧪 Jalankan Demo'** untuk mencoba.")

    st.markdown("---")

    # Status Perangkat Komputasi Hardware
    _, dev_info = get_optimal_device()
    st.markdown("#### ⚡ Status Komputasi Hardware & Model AI")
    if dev_info.get("type") == "GPU":
        st.success(f"🚀 **Akselerasi GPU Aktif:** {dev_info.get('desc')} • cuDNN Benchmark & TF32 Diaktifkan (Akselerasi Maksimal)")
        st.caption("💡 *Catatan Pemantauan GPU di Windows Task Manager:* Inferensi model AI (Deep Learning) berjalan di engine **Compute_0** / **CUDA**. Pada tab Performance Task Manager, klik drop-down grafis GPU dan pilih **Cuda** / **Compute_0** untuk memantau beban komputasi GPU yang sesungguhnya (bukan engine 3D).")
    elif dev_info.get("torch_cuda_missing"):
        hw_name = dev_info.get("hardware_gpu_name", "NVIDIA GPU")
        st.warning(
            f"💡 **Hardware GPU Terdeteksi di Komputer Anda: `{hw_name}`**, "
            "tetapi pustaka PyTorch yang aktif saat ini adalah versi CPU.\n\n"
            "Untuk mengaktifkan akselerasi GPU otomatis di laptop Anda, jalankan perintah ini di Command Prompt / Terminal:\n\n"
            f"```bash\n{dev_info.get('install_cuda_cmd')}\n```\n\n"
            "Setelah itu restart dashboard (`streamlit run app.py`), dan dashboard akan langsung menggunakan GPU Anda 100%!"
        )
        st.info(f"⚙️ **Status Komputasi Saat Ini:** Berjalan pada {dev_info.get('desc')} (Semua core CPU aktif).")
    else:
        st.info(f"⚙️ **Komputasi CPU:** {dev_info.get('desc')}")
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.info("⚡ **Cracks Seg:** `cracks/weights/best.pt` (<1mm, 1-3mm, >3mm)")
    with col_m2:
        st.info("🕳️ **Potholes Seg:** `potholes/weights/best.pt` (Deteksi & Mask Lubang)")
    with col_m3:
        st.info("📉 **Rutting Seg:** `rutting/weights/best.pt` (<1cm, 1-3cm, >3cm)")
    with col_m4:
        st.info("🛣️ **Pavement & GPS:** `perkerasan_cls` + `gps_sta_model/`")

    if st.session_state.preview_frames:
        st.markdown("#### 🖼️ Pratinjau Tangkapan Frame Resolusi Penuh")
        frame_select_idx = st.slider("Pilih Frame Sampel", 0, len(st.session_state.preview_frames) - 1, 0)
        selected_sample = st.session_state.preview_frames[frame_select_idx]

        col_img, col_det_info = st.columns([3, 1])
        with col_img:
            st.image(
                selected_sample["image"],
                caption=f"Frame #{selected_sample['frame_idx']} - STA {selected_sample['sta_m']:.1f} m",
                use_container_width=True
            )
        with col_det_info:
            st.metric("STA Posisi", f"{selected_sample['sta_m']:.1f} m")
            st.metric("Jenis Perkerasan", selected_sample["pavement"])
            st.metric("Kerusakan Terdeteksi", f"{selected_sample['det_count']} objek")
            st.caption("🟠 Kontur Poligon Retak • 🔴 Kontur Poligon Lubang • 🟣 Kontur Poligon Alur Rutting (Murni Segmentasi)")


# ------------------------------------------------------------------------------
# TAB 2: TABEL HASIL SDI (UTAMA) - 100% BEBAS DYNAMIC CHUNK ERROR
# ------------------------------------------------------------------------------
with tab_hasil:
    curr_road = st.session_state.road_info.get("road_name", "").strip()
    if curr_road:
        st.subheader(f"Tabel Rekapitulasi Kondisi Jalan - {curr_road}")
    else:
        st.subheader("Tabel Rekapitulasi Kondisi Jalan")

    segments = st.session_state.segments
    if not segments:
        st.info("💡 Data belum tersedia. Silakan jalankan proses video atau tekan tombol 'Jalankan Demo' di sidebar.")
    else:
        # Gunakan selectbox yang 100% stabil di semua browser (tanpa async chunk multiselect)
        col_tog1, col_tog2, col_tog3 = st.columns([2, 1, 1])
        with col_tog1:
            show_details = st.checkbox("🔍 Tampilkan Kolom Detail (% Luas Retak, Lebar Retak, Lubang/km, Alur, Flag)", value=False)
        with col_tog2:
            filter_cond_choice = st.selectbox(
                "Filter Kondisi",
                ["Semua Kondisi", "Baik", "Sedang", "Rusak Ringan", "Rusak Berat"],
                index=0
            )
        with col_tog3:
            all_paves = sorted(list(set(s.pavement_type for s in segments)))
            filter_pave_choice = st.selectbox(
                "Filter Perkerasan",
                ["Semua Perkerasan"] + all_paves,
                index=0
            )

        # Terapkan filter
        filtered_segments = segments
        if filter_cond_choice != "Semua Kondisi":
            filtered_segments = [s for s in filtered_segments if s.condition == filter_cond_choice]
        if filter_pave_choice != "Semua Perkerasan":
            filtered_segments = [s for s in filtered_segments if s.pavement_type == filter_pave_choice]

        # Siapkan DataFrame terstruktur
        df_table = segments_to_dataframe(filtered_segments, include_details=show_details)

        # Pewarnaan sel kolom Kondisi sesuai standar Bina Marga (Tabel 10)
        def color_condition(val):
            colors = {
                "Baik": "background-color: #D4EDDA; color: #155724; font-weight: bold;",
                "Sedang": "background-color: #FFF3CD; color: #856404; font-weight: bold;",
                "Rusak Ringan": "background-color: #FFE5D0; color: #B35300; font-weight: bold;",
                "Rusak Berat": "background-color: #F8D7DA; color: #721C24; font-weight: bold;",
            }
            return colors.get(val, "")

        # Tampilkan tabel data interaktif dengan pewarnaan kondisi
        try:
            styled_df = df_table.style.map(color_condition, subset=["Kondisi"])
            st.dataframe(styled_df, use_container_width=True, hide_index=True)
        except Exception:
            st.dataframe(df_table, use_container_width=True, hide_index=True)

        st.caption(f"Menampilkan {len(filtered_segments)} dari total {len(segments)} segmen.")

        with st.expander("📖 Panduan & Logika Rumus SDI (Tabel 9 & 10 PKRMS)"):
            st.markdown("""
            - **Level 1 (% Luas Retak):** Tidak ada $\\to 0$ ; $< 10\\% \\to 5$ ; $10 - 30\\% \\to 20$ ; $> 30\\% \\to 40$.
            - **Level 2 (Lebar Retak):** Rata-rata $> 3$ mm (atau $> 5$ mm) $\\to \\text{SDI}_2 = \\text{SDI}_1 \\times 2$ ; lainnya $\\text{SDI}_2 = \\text{SDI}_1$.
            - **Level 3 (Lubang/km):** Tidak ada $\\to +0$ ; $< 10$/km $\\to +15$ ; $10 - 50$/km $\\to +75$ ; $> 50$/km $\\to +225$.
            - **Level 4 (Alur/Rutting cm):** Tidak ada $\\to +0$ ; $< 1$ cm $\\to +2.5$ ; $1 - 3$ cm $\\to +10$ ; $> 3$ cm $\\to +20$.
            - **Klasifikasi Kondisi (Tabel 10):**
              - $\\text{SDI} < 50$ : **Baik** (🟢 Hijau)
              - $50 \\le \\text{SDI} < 100$ : **Sedang** (🟡 Kuning)
              - $100 \\le \\text{SDI} < 150$ : **Rusak Ringan** (🟠 Oranye)
              - $\\text{SDI} \\ge 150$ : **Rusak Berat** (🔴 Merah)
            """)


# ------------------------------------------------------------------------------
# TAB 3: VISUALISASI & KPI
# ------------------------------------------------------------------------------
with tab_vis:
    curr_road = st.session_state.road_info.get("road_name", "").strip()
    if curr_road:
        st.subheader(f"Ringkasan KPI & Analisis Kondisi - {curr_road}")
    else:
        st.subheader("Ringkasan KPI & Analisis Kondisi")

    segments = st.session_state.segments
    if not segments:
        st.info("💡 Belum ada data untuk divisualisasikan. Jalankan proses video atau demo terlebih dahulu.")
    else:
        total_length_m = sum(s.length_m for s in segments)
        weighted_sdi = (
            sum(s.sdi_value * s.length_m for s in segments) / total_length_m
            if total_length_m > 0 else 0.0
        )
        mantap_length_m = sum(s.length_m for s in segments if s.condition in ["Baik", "Sedang"])
        kemantapan_pct = (mantap_length_m / total_length_m * 100.0) if total_length_m > 0 else 0.0
        rusak_length_m = total_length_m - mantap_length_m

        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        with kpi1:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Panjang Total Otomatis</div>
                <div class="kpi-val">{total_length_m:.1f} m</div>
                <div class="kpi-note">{total_length_m/1000:.2f} km • {len(segments)} segmen</div>
            </div>
            """, unsafe_allow_html=True)
        with kpi2:
            st.markdown(f"""
            <div class="kpi-card" style="border-left-color: #2ECC71;">
                <div class="kpi-title">SDI Terbobot Panjang</div>
                <div class="kpi-val">{weighted_sdi:.1f}</div>
                <div class="kpi-note">Kategori: <strong>{classify_sdi_condition(weighted_sdi)['label']}</strong></div>
            </div>
            """, unsafe_allow_html=True)
        with kpi3:
            st.markdown(f"""
            <div class="kpi-card" style="border-left-color: #F1C40F;">
                <div class="kpi-title">Tingkat Kemantapan</div>
                <div class="kpi-val">{kemantapan_pct:.1f}%</div>
                <div class="kpi-note">Target DAK Prov &gt;75%, Kab &gt;65%</div>
            </div>
            """, unsafe_allow_html=True)
        with kpi4:
            st.markdown(f"""
            <div class="kpi-card" style="border-left-color: #E74C3C;">
                <div class="kpi-title">Segmen Kritis (Rusak)</div>
                <div class="kpi-val">{rusak_length_m:.1f} m</div>
                <div class="kpi-note">{sum(1 for s in segments if s.condition in ['Rusak Ringan', 'Rusak Berat'])} segmen butuh penanganan</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        st.subheader("📈 Profil Nilai SDI Sepanjang Ruas Jalan")
        chart_data = []
        for s in segments:
            center_sta = (s.sta_start + s.sta_end) / 2.0
            chart_data.append({
                "STA Label": f"{s.sta_start:.0f}-{s.sta_end:.0f}m",
                "Center STA": center_sta,
                "STA Start": s.sta_start,
                "STA End": s.sta_end,
                "Panjang": s.length_m,
                "Nilai SDI": s.sdi_value,
                "Kondisi": s.condition,
                "Perkerasan": s.pavement_type,
            })
        df_chart = pd.DataFrame(chart_data)

        fig_bar = px.bar(
            df_chart,
            x="STA Label",
            y="Nilai SDI",
            color="Kondisi",
            color_discrete_map={
                "Baik": "#2ECC71",
                "Sedang": "#F1C40F",
                "Rusak Ringan": "#E67E22",
                "Rusak Berat": "#E74C3C"
            },
            hover_data=["STA Start", "STA End", "Panjang", "Perkerasan"],
            title=f"Distribusi Nilai SDI per Segmen - {curr_road}" if curr_road else "Distribusi Nilai SDI per Segmen"
        )
        fig_bar.add_hline(y=50, line_dash="dash", line_color="#F1C40F", annotation_text="Batas Sedang (50)")
        fig_bar.add_hline(y=100, line_dash="dash", line_color="#E67E22", annotation_text="Batas Rusak Ringan (100)")
        fig_bar.add_hline(y=150, line_dash="dash", line_color="#E74C3C", annotation_text="Batas Rusak Berat (150)")
        fig_bar.update_layout(xaxis_title="Rentang STA (meter)", yaxis_title="Nilai SDI")
        st.plotly_chart(fig_bar, use_container_width=True)

        col_pie1, col_pie2 = st.columns(2)
        with col_pie1:
            st.subheader("🍰 Proporsi Kondisi Jalan (% Panjang)")
            cond_lengths = {}
            for s in segments:
                cond_lengths[s.condition] = cond_lengths.get(s.condition, 0.0) + s.length_m
            df_cond_pie = pd.DataFrame([
                {"Kondisi": k, "Panjang": v} for k, v in cond_lengths.items()
            ])
            fig_cond_pie = px.pie(
                df_cond_pie,
                values="Panjang",
                names="Kondisi",
                color="Kondisi",
                color_discrete_map={
                    "Baik": "#2ECC71",
                    "Sedang": "#F1C40F",
                    "Rusak Ringan": "#E67E22",
                    "Rusak Berat": "#E74C3C"
                },
                hole=0.4
            )
            st.plotly_chart(fig_cond_pie, use_container_width=True)

        with col_pie2:
            st.subheader("🧱 Proporsi Jenis Perkerasan (% Panjang)")
            pave_lengths = {}
            for s in segments:
                pave_lengths[s.pavement_type] = pave_lengths.get(s.pavement_type, 0.0) + s.length_m
            df_pave_pie = pd.DataFrame([
                {"Perkerasan": k, "Panjang": v} for k, v in pave_lengths.items()
            ])
            fig_pave_pie = px.pie(
                df_pave_pie,
                values="Panjang",
                names="Perkerasan",
                hole=0.4
            )
            st.plotly_chart(fig_pave_pie, use_container_width=True)


# ------------------------------------------------------------------------------
# TAB 4: PETA SPASIAL KONDISI JALAN
# ------------------------------------------------------------------------------
with tab_peta:
    curr_road = st.session_state.road_info.get("road_name", "").strip()
    if curr_road:
        st.subheader(f"Peta Spasial Segmen - {curr_road}")
    else:
        st.subheader("Peta Spasial Segmen")

    segments = st.session_state.segments
    if not segments:
        st.info("💡 Belum ada data spasial yang tersedia.")
    else:
        base_lat = -6.2088
        base_lon = 106.8456

        path_data = []
        for s in segments:
            lat_start = base_lat + (s.sta_start * 0.000009)
            lon_start = base_lon + (s.sta_start * 0.000009)
            lat_end = base_lat + (s.sta_end * 0.000009)
            lon_end = base_lon + (s.sta_end * 0.000009)

            color_rgb = {
                "Baik": [46, 204, 113, 220],
                "Sedang": [241, 196, 15, 220],
                "Rusak Ringan": [230, 126, 34, 220],
                "Rusak Berat": [231, 76, 60, 220],
            }.get(s.condition, [100, 100, 100, 200])

            path_data.append({
                "name": f"STA {s.sta_start:.0f}-{s.sta_end:.0f}m",
                "path": [[lon_start, lat_start], [lon_end, lat_end]],
                "color": color_rgb,
                "sdi": s.sdi_value,
                "condition": s.condition,
                "pavement": s.pavement_type,
                "recommendation": s.recommendation
            })

        view_state = pdk.ViewState(
            latitude=base_lat + (total_length_m * 0.0000045),
            longitude=base_lon + (total_length_m * 0.0000045),
            zoom=15,
            pitch=30
        )

        layer = pdk.Layer(
            "PathLayer",
            path_data,
            pickable=True,
            get_color="color",
            width_scale=20,
            width_min_pixels=6,
            get_path="path",
            get_width=5
        )

        st.pydeck_chart(
            pdk.Deck(
                layers=[layer],
                initial_view_state=view_state,
                tooltip={"text": "{name}\nSDI: {sdi}\nKondisi: {condition}\nPerkerasan: {pavement}\nSaran: {recommendation}"}
            )
        )
        st.caption("Legenda Warna Peta: 🟢 Hijau (Baik) • 🟡 Kuning (Sedang) • 🟠 Oranye (Rusak Ringan) • 🔴 Merah (Rusak Berat)")


# ------------------------------------------------------------------------------
# TAB 5: PANEL KOREKSI MANUAL
# ------------------------------------------------------------------------------
with tab_koreksi:
    st.subheader("Panel Koreksi Manual Segmen & Hitung Ulang")
    st.markdown(
        "Petugas survei dapat mengoreksi batas titik potong STA, jenis perkerasan, "
        "maupun memasukkan data kedalaman alur (rutting cm) hasil pengukuran lapangan secara langsung."
    )

    segments = st.session_state.segments
    if not segments:
        st.info("💡 Tidak ada data segmen untuk dikoreksi.")
    else:
        edit_records = []
        for idx, s in enumerate(segments):
            edit_records.append({
                "Index": idx,
                "Nama Ruas": s.road_name,
                "STA Awal (m)": s.sta_start,
                "STA Akhir (m)": s.sta_end,
                "Perkerasan": s.pavement_type,
                "Alur Rutting (cm)": s.rut_depth_cm,
                "Kondisi Saat Ini": s.condition,
                "SDI Saat Ini": s.sdi_value
            })

        df_editor_input = pd.DataFrame(edit_records)
        edited_df = st.data_editor(
            df_editor_input,
            key="manual_segment_editor",
            use_container_width=True,
            disabled=["Index", "Kondisi Saat Ini", "SDI Saat Ini"],
            column_config={
                "Perkerasan": st.column_config.SelectboxColumn(
                    "Perkerasan",
                    options=["Hotmix", "Beton", "Lapen", "Kerikil", "Tanah"],
                    required=True
                ),
                "Alur Rutting (cm)": st.column_config.NumberColumn(
                    "Alur Rutting (cm)",
                    min_value=0.0,
                    max_value=15.0,
                    step=0.5
                )
            }
        )

        col_kor1, col_kor2 = st.columns([1, 3])
        with col_kor1:
            recalculate_btn = st.button("🔄 Hitung Ulang SDI", type="primary")

        if recalculate_btn:
            overrides = []
            for _, row in edited_df.iterrows():
                overrides.append({
                    "index": int(row["Index"]),
                    "sta_start": float(row["STA Awal (m)"]),
                    "sta_end": float(row["STA Akhir (m)"]),
                    "pavement_type": str(row["Perkerasan"]),
                    "rut_depth_cm": float(row["Alur Rutting (cm)"])
                })

            updated_segs = apply_manual_corrections(segments, overrides)

            road_w = st.session_state.road_info.get("road_width_m", 6.0)
            crack_thresh = 3.0  # Standar Tabel 9 PKRMS (>3 mm)

            for seg in updated_segs:
                surface_area = max(1.0, seg.length_m * road_w)
                new_crack_pct = min(100.0, (seg.total_crack_area_m2 / surface_area) * 100.0)
                seg.crack_area_pct = round(new_crack_pct, 2)

                new_potholes_km = (seg.unique_potholes * 1000.0) / max(1.0, seg.length_m)
                seg.potholes_per_km = round(new_potholes_km, 2)

                sdi_res = calculate_sdi(
                    crack_area_pct=seg.crack_area_pct,
                    avg_crack_width_mm=seg.avg_crack_width_mm,
                    potholes_per_km=seg.potholes_per_km,
                    rut_depth_cm=seg.rut_depth_cm,
                    crack_width_threshold_mm=crack_thresh,
                    config=CONFIG
                )

                rec_txt, rec_note = get_recommendation(
                    condition=sdi_res.condition,
                    pavement_type=seg.pavement_type,
                    config=CONFIG
                )
                seg.sdi_value = sdi_res.final_sdi
                seg.condition = sdi_res.condition
                seg.color = sdi_res.color
                seg.recommendation = f"{rec_txt} ({rec_note})" if rec_note else rec_txt

            st.session_state.segments = updated_segs
            st.success("✅ Nilai SDI berhasil dihitung ulang dengan parameter koreksi pengguna!")
            st.rerun()


# ------------------------------------------------------------------------------
# TAB 6: EKSPOR LAPORAN
# ------------------------------------------------------------------------------
with tab_export:
    curr_road = st.session_state.road_info.get("road_name", "").strip()
    if curr_road:
        st.subheader(f"Ekspor Data & Laporan Hasil Survei - {curr_road}")
    else:
        st.subheader("Ekspor Data & Laporan Hasil Survei")

    segments = st.session_state.segments
    if not segments:
        st.info("💡 Belum ada data untuk diekspor.")
    else:
        st.markdown(
            "Unduh laporan lengkap hasil evaluasi survei kondisi jalan berbasis SDI "
            "dalam berbagai format untuk pelaporan Dinas PUPR / Ditjen Bina Marga:"
        )

        col_ex1, col_ex2, col_ex3, col_ex4 = st.columns(4)

        with col_ex1:
            csv_data = export_to_csv(segments)
            st.download_button(
                label="📄 Unduh Data CSV",
                data=csv_data,
                file_name=f"survei_sdi_{st.session_state.road_info.get('road_name', 'jalan')}.csv",
                mime="text/csv",
                use_container_width=True
            )

        with col_ex2:
            excel_bytes = export_to_excel(segments, st.session_state.road_info)
            st.download_button(
                label="📊 Unduh Excel (.xlsx)",
                data=excel_bytes,
                file_name=f"laporan_sdi_{st.session_state.road_info.get('road_name', 'jalan')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )

        with col_ex3:
            html_report = export_to_html_report(segments, st.session_state.road_info)
            st.download_button(
                label="📑 Unduh Cetak (HTML)",
                data=html_report,
                file_name=f"laporan_resmi_sdi_{st.session_state.road_info.get('road_name', 'jalan')}.html",
                mime="text/html",
                use_container_width=True
            )

        with col_ex4:
            vid_p = st.session_state.annotated_video_path
            if vid_p and os.path.exists(vid_p):
                sz_mb = os.path.getsize(vid_p) / (1024 * 1024)
                abs_vid_p = os.path.abspath(vid_p)
                if sz_mb <= 150:
                    try:
                        with open(vid_p, "rb") as vf:
                            vid_dl_bytes = vf.read()
                        st.download_button(
                            label="🎬 Unduh Video (.mp4)",
                            data=vid_dl_bytes,
                            file_name=f"video_inferensi_{st.session_state.road_info.get('road_name', 'jalan')}.mp4",
                            mime="video/mp4",
                            use_container_width=True
                        )
                    except Exception:
                        if st.button("📂 Buka Video di Explorer", use_container_width=True):
                            subprocess.Popen(f'explorer /select,"{abs_vid_p}"')
                else:
                    if st.button("📂 Buka Video di Explorer", use_container_width=True):
                        subprocess.Popen(f'explorer /select,"{abs_vid_p}"')
            else:
                st.button("🎬 Unduh Video (.mp4)", disabled=True, use_container_width=True)

        st.markdown("---")
        with st.expander("👁️ Pratinjau Laporan Ringkas"):
            st.components.v1.html(html_report, height=600, scrolling=True)
