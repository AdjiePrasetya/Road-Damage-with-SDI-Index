# 🛣️ Dashboard Survei Kondisi Jalan Berbasis SDI (PKRMS)

Aplikasi Dashboard Streamlit untuk evaluasi survei kondisi jalan berbasis **Surface Distress Index (SDI)** sesuai standar resmi **Manual PKRMS** (Surat Edaran Direktur Jenderal Bina Marga No. 22/SE/Db/2021, Manual No. 04/M/BM/2021, Bab 4.2.5, Tabel 9 & 10).

---

## 📌 Gambaran Umum Alur Sistem

```
[ Video Survei Jalan ]
         │
         ▼
[ Sampling Frame & Ekstraksi ]
         │
         ├───► [ RoadDamageDetector ] ──► BBox/Mask, Kategori SDI (Crack/Pothole), Area m2, Lebar mm
         │            │
         │            ▼
         │     [ SimpleDamageTracker ] ──► Deduplikasi objek antar-frame (cegah double counting)
         │
         ├───► [ PavementClassifier ] ──► Klasifikasi Jenis Perkerasan (Hotmix, Beton, Lapen, dll)
         │            │
         │            ▼
         │     [ Majority / Median Filter ] ──► Penghilangan flicker transisi perkerasan
         │
         └───► [ StaEstimator ] ──────► Estimasi Posisi STA (jarak meter) & Koordinat GPS
                      │
                      ▼
        [ Logika Segmentasi PKRMS ]
         • Blok dasar tiap 100 m
         • Pemotongan segmen saat jenis perkerasan berubah (misal 0-60m Hotmix, 60-100m Beton)
         • Penggabungan segmen pendek (< 10 m) ke tetangga + flag
                      │
                      ▼
           [ Perhitungan SDI Bertingkat ]
         • Level 1: % Luas Retak (SDI1: 0 / 5 / 20 / 40)
         • Level 2: Lebar Retak (SDI2 = SDI1 * 2 jika > 3mm)
         • Level 3: Lubang / km (SDI3 = SDI2 + 0 / 15 / 75 / 225)
         • Level 4: Alur Bekas Roda cm (SDI4 = SDI3 + 0 / 2.5 / 10 / 20)
                      │
                      ▼
         [ Klasifikasi Kondisi (Tabel 10) ]
         • < 50         : Baik (Hijau)
         • 50 - < 100   : Sedang (Kuning)
         • 100 - < 150  : Rusak Ringan (Oranye)
         • >= 150       : Rusak Berat (Merah)
                      │
                      ▼
         [ Saran Penanganan & Ekspor ]
         • Rekomendasi aspal vs beton
         • Ekspor ke CSV, Excel Formatted (.xlsx), dan Laporan HTML/Cetak
```

---

## 📂 Struktur Repositori

```
dashboard_sdi/
├── .streamlit/
│   └── config.toml             # Konfigurasi Streamlit (maxUploadSize = 1024 MB)
├── config.yaml                 # Konfigurasi terpusat (mapping kelas, formula SDI, ambang batas, saran)
├── app.py                      # Aplikasi utama Streamlit (Multi-tab)
├── requirements.txt            # Dependensi Python
├── packages.txt                # Dependensi Linux untuk Streamlit Cloud
├── core/
│   ├── __init__.py
│   ├── sdi.py                  # Rumus perhitungan SDI Level 1-4 & Klasifikasi Kondisi
│   ├── segmentation.py         # Pembagian segmen 100m, split perkerasan, merge segmen pendek
│   ├── recommendation.py       # Rule-based saran teknis perbaikan (aspal & rigid beton)
│   ├── video.py                # Pipeline ekstraksi frame, tracking/deduplikasi, anotasi visual
│   ├── exporter.py             # Generator export CSV, Excel rapi, & Laporan HTML
│   └── demo_data.py            # Generator data survei sintetis untuk pengujian instan
├── models/
│   ├── __init__.py
│   ├── damage_detector.py      # Adapter model YOLO Deteksi Kerusakan (RDD2022)
│   ├── pavement_classifier.py  # Adapter model YOLO Klasifikasi Perkerasan
│   └── sta_estimator.py        # Adapter estimasi STA (Speed-time / OCR overlay / Custom)
├── tests/
│   ├── test_sdi.py             # Unit tests perhitungan batas rumus SDI Tabel 9 & 10
│   └── test_segmentation.py    # Unit tests logika segmentasi & koreksi manual
├── gps_sta_model/              # Folder model/data GPS & STA
├── perkerasan_clf_models/      # Folder model klasifikasi perkerasan
└── road_damage_models/         # Folder model deteksi kerusakan jalan
```

---

## 🔍 Temuan Inspeksi Model & Folder

1. **`road_damage_models/`**:
   - Model terlatih: `runs/detect/rdd_baseline/run1/weights/best.pt`
   - Framework: **Ultralytics YOLO (Detection)**
   - Kelas (RDD2022):
     - `longitudinal crack` $\to$ crack
     - `transverse crack` $\to$ crack
     - `alligator crack` $\to$ crack
     - `other corruption` $\to$ other
     - `Pothole` $\to$ pothole
2. **`perkerasan_clf_models/`**:
   - Model terlatih: `runs/perkerasan_cls/weights/best.pt`
   - Framework: **Ultralytics YOLO (Classification / yolo11s-cls)**
   - Kelas (5 tipe):
     - `hotmix` $\to$ Hotmix (Aspal)
     - `lapen` $\to$ Lapen (Aspal Penetrasi)
     - `perkerasan_beton` $\to$ Beton (Rigid Pavement)
     - `kerikil` $\to$ Kerikil
     - `tanah_belum_tembus` $\to$ Tanah
3. **`gps_sta_model/`**:
   - Checkpoint default `weights/yolo26n.pt` berisi weights COCO umum.
   - Sesuai panduan arsitektur, adapter `StaEstimator` mendukung 3 mode:
     - `speed_time` (default & andal): menghitung STA dari waktu frame dan kecepatan survei rata-rata kendaraan ($v \times t$).
     - `ocr_overlay`: membaca HUD teks koordinat/STA yang tercetak pada video dashcam menggunakan EasyOCR.
     - `custom_yolo`: placeholder untuk model kustom deteksi patok KM.

---

## ⚙️ Bagian yang Perlu Dikonfirmasi Pengguna (`TODO (USER)`)

Semua konfigurasi berikut berada di `config.yaml` dan dapat disesuaikan tanpa mengubah kode program:

1. **Mapping Kelas Model (`config.yaml` $\to$ `class_mapping`)**:
   - Pastikan label kelas model RDD2022 Anda dipetakan ke kategori yang diinginkan (`crack`, `pothole`, `other`).
2. **Kalibrasi Pixel ke Meter (`config.yaml` $\to$ `calibration.meter_per_pixel`)**:
   - Tentukan rasio meter per pixel kamera survei Anda (default: `0.005` m/pixel).
3. **Faktor Luas Retak dari Bounding Box (`config.yaml` $\to$ `calibration.bbox_crack_area_factor`)**:
   - Karena model RDD2022 mendeteksi kotak pembatas (bounding box), luas retak nyata bukan luas seluruh kotak. Default diset ke `0.20` (20% luas kotak adalah retak nyata).
4. **Ambang Lebar Retak (`config.yaml` $\to$ `sdi_table_9.crack_width.threshold_mm`)**:
   - Tabel 9 Manual PKRMS mencantumkan `> 3 mm`, sedangkan narasi teks halaman 38 menyebutkan `> 5 mm`. Pilihan ini tersedia di antarmuka sidebar pengguna (default: 3 mm).

---

## 🚀 Panduan Clone Repo & Instalasi Lokal (Auto-Detect GPU & CPU)

Aplikasi ini didesain **100% otomatis mendeteksi hardware komputer lokal pengguna**:
- **Jika komputer/laptop memiliki GPU NVIDIA (GeForce GTX/RTX, Quadro, Tesla):** Dashboard otomatis mendeteksi nama GPU, kapasitas VRAM, dan mengaktifkan akselerasi GPU (Tensor Cores TF32 & cuDNN) secara maksimal.
- **Jika komputer berbasis Apple Silicon (Mac M1/M2/M3/M4):** Dashboard otomatis menggunakan akselerasi Metal (MPS).
- **Jika komputer tanpa GPU (hanya CPU):** Dashboard otomatis menggunakan CPU multi-threading maksimal ke seluruh core prosesor tanpa ada error.

### 1. Clone Repositori
```bash
git clone https://github.com/<username>/dashboard_sdi.git
cd dashboard_sdi
```

### 2. Instalasi Dependensi

#### Opsi A: Jika Laptop / PC Memiliki GPU NVIDIA (Direkomendasikan untuk Akselerasi Maksimal)
Jalankan file script otomatis di Windows:
```cmd
install_gpu.bat
```
Atau jalankan manual lewat terminal:
```bash
# 1. Pasang PyTorch berkemampuan CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 2. Pasang dependensi dashboard lainnya
pip install -r requirements.txt
```

#### Opsi B: Jika Komputer Hanya Menggunakan CPU (Tanpa GPU Dedicated)
```bash
pip install -r requirements.txt
```

### 3. Menjalankan Unit Tests (Opsional)
Uji validitas batas perhitungan rumus SDI dan segmentasi jalan:
```bash
python -m unittest discover tests/ -v
```

### 4. Menjalankan Dashboard Streamlit
```bash
streamlit run app.py
```
Akses dashboard di browser melalui `http://localhost:8501`.

> 💡 **Deteksi Otomatis & Diagnosis GPU Terintegrasi:**
> Jika pengguna memiliki GPU NVIDIA pada laptopnya tetapi keliru menginstal PyTorch versi CPU, Dashboard akan langsung mendeteksi keberadaan kartu grafis tersebut dan memunculkan petunjuk instalasi 1-baris langsung di antarmuka web untuk mengaktifkan GPU.

---

## 🧪 Mode Demo Tanpa Video

Jika Anda ingin menguji seluruh tampilan antarmuka, grafik analitik, tabel hasil SDI, dan peta spasial tanpa mengunggah video besar:
1. Buka dashboard di browser.
2. Di sidebar sebelah kiri, klik tombol **"🧪 Jalankan Demo"**.
3. Dashboard akan langsung terisi dengan 5 segmen contoh realistis termasuk kasus pemotongan perkerasan (0 - 60 m Hotmix dan 60 - 100 m Beton).

---

## ☁️ Panduan Deploy ke Streamlit Community Cloud

Saat mendeploy ke **Streamlit Community Cloud**:
1. **Resource Batas**: Streamlit Cloud gratis hanya menyediakan CPU (tanpa GPU) dan RAM terbatas (~1-2 GB).
2. **Optimasi yang Diterapkan**:
   - Menggunakan `opencv-python-headless` (bukan OpenCV standar dengan X11).
   - Dependensi sistem Linux disediakan di `packages.txt` (`libgl1`, `ffmpeg`).
   - Ekstraksi frame dilakukan secara bertahap (generator) dengan parameter sampling frame (`frame_step`), sehingga tidak membebani memori RAM.
   - Pemuatan model dibungkus dengan `@st.cache_resource` agar hanya dimuat satu kali saat startup.
3. **Batas Ukuran File GitHub (100 MB)**:
   - File model PyTorch `.pt` yang berukuran $> 100$ MB tidak boleh dipush langsung via Git standar.
   - Gunakan **Git LFS** (*Large File Storage*) atau unduh bobot model dari Google Drive / Hugging Face / S3 secara dinamis saat inisialisasi aplikasi jika file melebihi 100 MB.

