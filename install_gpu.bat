@echo off
chcp 65001 >nul
echo ==============================================================================
echo  🚀 INSTALASI OTOMATIS DASHBOARD SURVEI SDI - MODE GPU NVIDIA (CUDA)
echo ==============================================================================
echo.
echo Memeriksa Python...
python --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Python tidak ditemukan. Pastikan Python 3.10 atau 3.11 sudah terpasang dan ditambahkan ke PATH.
    pause
    exit /b 1
)

echo [1/3] Menginstal PyTorch dengan dukungan CUDA (NVIDIA GPU)...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

echo.
echo [2/3] Menginstal seluruh dependensi pendukung...
pip install -r requirements.txt

echo.
echo [3/3] Memverifikasi deteksi GPU...
python -c "import torch; print('--------------------------------------------------'); print('Deteksi GPU CUDA:', 'SUKSES' if torch.cuda.is_available() else 'BELUM AKTIF (Mode CPU)'); print('Perangkat GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'); print('--------------------------------------------------')"

echo.
echo ==============================================================================
echo  ✅ Selesai! Dashboard siap dijalankan dengan perintah:
echo     streamlit run app.py
echo ==============================================================================
pause

