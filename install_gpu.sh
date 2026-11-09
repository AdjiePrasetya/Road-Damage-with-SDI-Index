#!/bin/bash
# ==============================================================================
# 🚀 INSTALASI OTOMATIS DASHBOARD SURVEI SDI - MODE GPU NVIDIA (CUDA) / CPU
# ==============================================================================

echo "Memeriksa Python..."
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] python3 tidak ditemukan. Silakan pasang Python 3.10 atau 3.11."
    exit 1
fi

echo "[1/3] Menginstal PyTorch dengan dukungan CUDA (NVIDIA GPU)..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

echo "[2/3] Menginstal dependensi..."
pip install -r requirements.txt

echo "[3/3] Memverifikasi deteksi perangkat komputasi..."
python3 -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

echo "✅ Selesai! Jalankan dashboard dengan: streamlit run app.py"

