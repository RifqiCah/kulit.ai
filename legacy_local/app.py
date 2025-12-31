from fastapi import FastAPI, File, UploadFile
import torch
import torch.nn as nn
from torchvision import models, transforms
import sys
import os
import cv2
import numpy as np

# --- SETUP PATH ---
# Supaya bisa import dari folder src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

app = FastAPI(
    title="Kulit.ai API (Real Model)",
    description="Dermatology Analysis System powered by EfficientNetV2 (Real Inference)",
    version="2.0.0"
)

# --- 1. LOAD MODEL ASLI (Bukan Simulasi) ---
print("🧠 Loading REAL EfficientNetV2 Model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Arsitektur
model = models.efficientnet_v2_s(weights=None) 
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 3) # 3 Kelas

# Load Weights Hasil Training Kamu
# Pastikan file ini ada! (Hasil dari train_local.py atau download dari S3)
model_path = os.path.join(os.path.dirname(__file__), 'model.pth')

if os.path.exists(model_path):
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("✅ Model weights loaded successfully! System Ready.")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
else:
    print("⚠️ WARNING: File 'model.pth' tidak ditemukan di folder legacy_local!")
    print("⚠️ Aplikasi akan error saat prediksi. Jalankan train_local.py dulu!")

model = model.to(device)
model.eval()

# Label Kelas (Harus urut abjad sesuai folder: Dry, Normal, Oily)
LABELS = ['Dry', 'Normal', 'Oily']

# Fungsi Preprocess (Kita taruh sini biar mandiri dan aman)
def preprocess_image(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(img).unsqueeze(0)

@app.get("/")
def home():
    return {"message": "Kulit.ai Real-Inference System is Online 🚀"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    print(f"📸 Receiving image: {file.filename}")
    image_bytes = await file.read()
    
    try:
        # Preprocess
        tensor = preprocess_image(image_bytes).to(device)
        
        # --- PREDIKSI REAL (Math Calculation) ---
        with torch.no_grad():
            outputs = model(tensor)
            # Hitung probabilitas (%)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            
            # Ambil nilai tertinggi
            confidence, predicted_idx = torch.max(probabilities, 1)
        
        class_name = LABELS[predicted_idx.item()]
        conf_score = confidence.item() * 100
        
        # Ambil detail probabilitas semua kelas (opsional, biar keren)
        all_probs = {LABELS[i]: f"{probabilities[0][i].item()*100:.2f}%" for i in range(3)}

        return {
            "filename": file.filename,
            "prediction": class_name,
            "confidence": f"{conf_score:.2f}%",
            "details": all_probs,
            "status": "success (Real Inference)"
        }

    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    # Host 0.0.0.0 biar bisa diakses dari HP kalau satu wifi
    uvicorn.run(app, host="0.0.0.0", port=8000)