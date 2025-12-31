# 1. Pakai Base Image Python 3.10 (Versi Slim biar ringan)
FROM python:3.10-slim

# 2. Set folder kerja di dalam container
WORKDIR /app

# 3. Install System Dependencies untuk OpenCV
# (Wajib di Linux Slim, kalau gak nanti error "libGL.so missing")
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy requirements dan install library Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy Source Code & Aplikasi
# Folder 'src' dicopy agar bisa di-import
COPY src/ src/
# File aplikasi utama dicopy ke root /app
COPY legacy_local/app.py .

# 6. COPY MODEL ASLI (PENTING!)
# Kita masukkan file otak model.pth ke dalam container
# Pastikan file 'model.pth' sudah ada di folder legacy_local kamu!
COPY legacy_local/model.pth .

# 7. Perintah untuk menjalankan aplikasi saat container nyala
# Port 80 adalah port standar web
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]