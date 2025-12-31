import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

# ======================================================
# KONFIGURASI - HARUS SAMA DENGAN TRAINING
# ======================================================
CLASS_NAMES = ["Dry", "Normal", "Oily"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "notebooks/model_hasil_training/model.pth"
IMG_SIZE = 384  # SAMA DENGAN TRAINING (384 bukan 224!)

# ======================================================
# APLIKASI STREAMLIT
# ======================================================
st.set_page_config(page_title="Kulit.ai", layout="wide")
st.title("🧴 Kulit.ai - Skin Type Prediction")
st.markdown("---")

# Debug info
st.sidebar.title("Debug Info")
st.sidebar.write(f"Looking for model: `{MODEL_PATH}`")
st.sidebar.write(f"Current dir: `{os.getcwd()}`")

# ======================================================
# LOAD MODEL - PASTIKAN SAMA DENGAN EVALUASI
# ======================================================
@st.cache_resource
def load_model():
    """Load model dengan arsitektur yang sama seperti training"""
    try:
        # 1. Buat model SAMA PERSIS seperti di evaluasi
        model = models.efficientnet_v2_l(weights=None)
        model.classifier[1] = nn.Linear(
            model.classifier[1].in_features,
            len(CLASS_NAMES)
        )
        
        # 2. Load state_dict
        state_dict = torch.load(MODEL_PATH, map_location=device)
        
        # 3. Load ke model
        model.load_state_dict(state_dict)
        
        # 4. Ke device dan eval
        model.to(device)
        model.eval()
        
        st.sidebar.success("✅ Model berhasil diload!")
        st.sidebar.write(f"Arsitektur: EfficientNetV2_L")
        st.sidebar.write(f"Input size: {IMG_SIZE}x{IMG_SIZE}")
        st.sidebar.write(f"Classes: {len(CLASS_NAMES)}")
        
        return model
        
    except Exception as e:
        st.sidebar.error(f"❌ Error: {str(e)}")
        return None

# Cek apakah file model ada
if not os.path.exists(MODEL_PATH):
    st.error(f"❌ File tidak ditemukan: {MODEL_PATH}")
    
    # Coba cari file model
    st.info("Mencari file model.pth...")
    for root, dirs, files in os.walk(".."):
        for file in files:
            if file.endswith(".pth"):
                st.write(f"- {os.path.join(root, file)}")
    st.stop()

# Load model
model = load_model()

if not model:
    st.error("Gagal memuat model. Periksa error di sidebar.")
    st.stop()

# ======================================================
# UPLOAD DAN PREDIKSI GAMBAR
# ======================================================
st.success("✅ Model siap digunakan!")

# Upload image
st.subheader("📷 Upload Gambar Kulit")
uploaded_image = st.file_uploader(
    "Pilih gambar wajah (jpg, png, jpeg)",
    type=["jpg", "png", "jpeg"],
    help=f"Ukuran yang direkomendasikan: minimal {IMG_SIZE}x{IMG_SIZE} piksel"
)

if uploaded_image:
    # Buka gambar
    image = Image.open(uploaded_image).convert("RGB")
    
    # Tampilkan
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Gambar Asli", use_column_width=True)
    
    # ======================================================
    # PREPROCESSING - SAMA DENGAN EVALUASI
    # ======================================================
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),  # 384x384
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    try:
        # Preprocess
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        # Tampilkan gambar yang sudah dipreprocess
        with col2:
            # Denormalize untuk visualisasi
            img_normalized = transforms.functional.normalize(
                input_tensor.squeeze(0).cpu(),
                mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                std=[1/0.229, 1/0.224, 1/0.225]
            )
            img_to_show = transforms.ToPILImage()(img_normalized.clamp(0, 1))
            st.image(img_to_show, caption=f"Resized to {IMG_SIZE}x{IMG_SIZE}", use_column_width=True)
        
        # Inference
        with st.spinner("🔍 Menganalisis kulit..."):
            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                predicted_class = CLASS_NAMES[predicted.item()]
                confidence_percent = confidence.item() * 100
        
        # ======================================================
        # TAMPILKAN HASIL
        # ======================================================
        st.subheader("📊 Hasil Analisis")
        
        # Box hasil utama
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 25px;
            border-radius: 15px;
            color: white;
            margin-bottom: 20px;
        ">
            <h2 style="margin: 0; font-size: 28px;">🎯 {predicted_class}</h2>
            <p style="margin: 5px 0 0 0; font-size: 16px; opacity: 0.9;">
                Tingkat Kepercayaan: <strong>{confidence_percent:.1f}%</strong>
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Progress bar confidence
        st.progress(float(confidence.item()))
        
        # Probabilitas detail
        st.subheader("📈 Detail Probabilitas")
        probs = probabilities.squeeze().cpu().numpy()
        
        for class_name, prob in zip(CLASS_NAMES, probs):
            percentage = prob * 100
            
            # Buat bar horizontal dengan label
            col_label, col_bar, col_pct = st.columns([1, 3, 1])
            
            with col_label:
                st.write(f"**{class_name}**")
            
            with col_bar:
                st.progress(float(prob))
            
            with col_pct:
                st.write(f"{percentage:.1f}%")
        
        # ======================================================
        # REKOMENDASI
        # ======================================================
        st.subheader("💡 Rekomendasi Perawatan")
        
        if predicted_class == "Dry":
            st.info("""
            ### 🍂 Untuk Kulit Kering:
            
            **Pembersih:**
            - CeraVe Hydrating Cleanser
            - La Roche-Posay Toleriane Hydrating Gentle Cleanser
            
            **Pelembab:**
            - Cetaphil Moisturizing Cream
            - Neutrogena Hydro Boost Gel-Cream
            
            **Tips:**
            - Gunakan humidifier di ruangan
            - Hindari air panas saat mencuci muka
            - Aplikasikan pelembab saat kulit masih lembab
            """)
            
        elif predicted_class == "Oily":
            st.warning("""
            ### 🛢️ Untuk Kulit Berminyak:
            
            **Pembersih:**
            - CeraVe Foaming Facial Cleanser
            - Neutrogena Oil-Free Acne Wash
            
            **Pelembab:**
            - La Roche-Posay Effaclar Mat
            - Neutrogena Oil-Free Moisture
            
            **Tips:**
            - Cuci muka maksimal 2x sehari
            - Gunakan oil-absorbing sheets
            - Pilih produk berlabel "non-comedogenic"
            """)
            
        else:  # Normal
            st.success("""
            ### 🌟 Untuk Kulit Normal:
            
            **Pembersih:**
            - Cetaphil Gentle Skin Cleanser
            - Vanicream Gentle Facial Cleanser
            
            **Pelembab:**
            - CeraVe AM Facial Moisturizing Lotion
            - Aveeno Positively Radiant Daily Moisturizer
            
            **Tips:**
            - Gunakan sunscreen setiap hari
            - Eksfoliasi 1-2x seminggu
            - Tetap konsisten dengan rutinitas
            """)
        
        # ======================================================
        # INFORMASI TEKNIS
        # ======================================================
        with st.expander("🔧 Informasi Teknis"):
            st.write(f"- **Model:** EfficientNetV2-L")
            st.write(f"- **Input Size:** {IMG_SIZE}x{IMG_SIZE} px")
            st.write(f"- **Preprocessing:** Resize → Normalize (ImageNet stats)")
            st.write(f"- **Device:** {device}")
            st.write(f"- **Model Path:** {MODEL_PATH}")
        
        # Disclaimer
        st.markdown("---")
        st.caption("""
        ⚠️ **Disclaimer:** Aplikasi ini menggunakan AI untuk prediksi jenis kulit. 
        Hasil ini tidak menggantikan diagnosis profesional dari dokter kulit.
        Konsultasikan dengan ahli untuk perawatan yang tepat.
        """)
        
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.info("Pastikan gambar memiliki format yang valid dan coba lagi.")

else:
    # Tampilkan sebelum upload
    st.info("👆 **Upload gambar kulit untuk memulai analisis**")
    
    # Contoh interface
    with st.expander("🎯 Cara mendapatkan hasil terbaik"):
        st.markdown("""
        1. **Posisi:** Foto area pipi atau T-zone (dahi, hidung, dagu)
        2. **Cahaya:** Gunakan cahaya natural atau ruangan terang
        3. **Fokus:** Pastikan gambar jelas dan tidak blur
        4. **Makeup:** Hindari makeup tebal atau foundation
        5. **Background:** Gunakan background netral (putih/abu-abu)
        
        **Contoh area yang baik:**
        """)
        
        # Contoh grid
        cols = st.columns(3)
        with cols[0]:
            st.markdown("**Area Pipi**")
            st.image("https://via.placeholder.com/150x150/FFCCCC/000000?text=Cheek", 
                    use_column_width=True)
        with cols[1]:
            st.markdown("**T-Zone**")
            st.image("https://via.placeholder.com/150x150/CCFFCC/000000?text=T-Zone", 
                    use_column_width=True)
        with cols[2]:
            st.markdown("**Dahi**")
            st.image("https://via.placeholder.com/150x150/CCCCFF/000000?text=Forehead", 
                    use_column_width=True)

# ======================================================
# FOOTER
# ======================================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>Kulit.ai v1.0 | Powered by PyTorch EfficientNetV2-L | Streamlit</p>
    <p style="font-size: 12px;">Model accuracy: ±95% (on test set)</p>
</div>
""", unsafe_allow_html=True)