import streamlit as st
from PIL import Image
import boto3
import json
import os
import io
import base64
from dotenv import load_dotenv
from botocore.config import Config
import time

# ======================================================
# KONFIGURASI
# ======================================================
st.set_page_config(
    page_title="Kulit.ai - Skin Analysis",
    page_icon="🧴",
    layout="wide"
)

load_dotenv()

ENDPOINT_NAME = os.getenv("SAGEMAKER_ENDPOINT", "kulitai-gpu-prod")
REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
LAMBDA_FUNCTION_NAME = os.getenv("LAMBDA_FUNCTION_NAME", "")

# ======================================================
# FUNGSI HELPER
# ======================================================
def display_results(result):
    """Display analysis results"""
    st.success("✅ Analysis Complete!")
    
    # Extract prediction
    prediction = result.get("predicted_class", result.get("prediction", "Unknown"))
    confidence = result.get("confidence", 0)
    
    if isinstance(confidence, float) and confidence <= 1.0:
        confidence_pct = confidence * 100
    else:
        confidence_pct = confidence
    
    # Display main result
    col1, col2 = st.columns(2)
    
    with col1:
        # Color-coded result
        if "dry" in str(prediction).lower():
            st.markdown("""
            <div style='background-color: #FFF3CD; padding: 20px; border-radius: 10px;'>
            <h2 style='color: #856404;'>🍂 Dry Skin</h2>
            <p>Your skin needs extra hydration</p>
            </div>
            """, unsafe_allow_html=True)
        elif "oily" in str(prediction).lower():
            st.markdown("""
            <div style='background-color: #F8D7DA; padding: 20px; border-radius: 10px;'>
            <h2 style='color: #721C24;'>🛢️ Oily Skin</h2>
            <p>Your skin produces excess oil</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style='background-color: #D1ECF1; padding: 20px; border-radius: 10px;'>
            <h2 style='color: #0C5460;'>🌟 Normal Skin</h2>
            <p>Your skin is well-balanced</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.metric("Confidence", f"{confidence_pct:.1f}%")
        st.progress(float(min(confidence_pct/100, 1.0)))
    
    with col2:
        # Show probabilities
        st.markdown("### 📊 Probabilities")
        probs = result.get("probabilities", {})
        
        if isinstance(probs, list):
            labels = ["Dry", "Normal", "Oily"]
            probs = {labels[i]: probs[i] for i in range(min(len(labels), len(probs)))}
        
        for skin_type, prob in probs.items():
            if isinstance(prob, float) and prob <= 1.0:
                prob_pct = prob * 100
            else:
                prob_pct = prob
            
            st.write(f"**{skin_type}**")
            st.progress(float(min(prob_pct/100, 1.0)))
            st.caption(f"{prob_pct:.1f}%")
    
    # Recommendations
    st.markdown("---")
    st.markdown("### 💡 Recommendations")
    
    # Quick recommendations based on skin type
    if "dry" in str(prediction).lower():
        st.info("""
        **For Dry Skin:**
        - Use gentle, hydrating cleansers
        - Apply moisturizer on damp skin
        - Avoid hot water when washing face
        - Consider using a humidifier
        """)
    elif "oily" in str(prediction).lower():
        st.warning("""
        **For Oily Skin:**
        - Use oil-free, non-comedogenic products
        - Cleanse twice daily with gentle cleanser
        - Use blotting papers throughout the day
        - Avoid touching your face frequently
        """)
    else:
        st.success("""
        **For Normal Skin:**
        - Maintain consistent skincare routine
        - Use sunscreen daily
        - Exfoliate 1-2 times per week
        - Stay hydrated and eat balanced diet
        """)
    
    # Raw response for debugging
    with st.expander("🔍 View Raw Response"):
        st.json(result)

def optimize_image_for_inference(image_bytes, target_size=(384, 384)):
    """Optimalisasi gambar untuk inferensi cepat"""
    try:
        # Buka gambar
        img = Image.open(io.BytesIO(image_bytes))
        
        # Convert ke RGB
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize ke ukuran target model
        img = img.resize(target_size, Image.Resampling.LANCZOS)
        
        # Konversi ke bytes dengan kompresi optimal
        buffer = io.BytesIO()
        
        # Pilih format berdasarkan ukuran
        if len(image_bytes) > 500000:  # > 500KB
            # Kompresi lebih agresif
            img.save(buffer, format='JPEG', quality=70, optimize=True)
        else:
            img.save(buffer, format='JPEG', quality=85, optimize=True)
        
        return buffer.getvalue()
        
    except Exception as e:
        st.error(f"Error optimizing image: {str(e)}")
        return image_bytes  # Fallback ke original

def invoke_sagemaker_with_retry(runtime_client, image_bytes, max_retries=2):
    """Invoke SageMaker endpoint dengan retry mechanism"""
    
    # Optimize image terlebih dahulu
    optimized_image = optimize_image_for_inference(image_bytes)
    
    for attempt in range(max_retries):
        try:
            # Tambah delay untuk cold start pada retry
            if attempt > 0:
                time.sleep(5)  # Tunggu 5 detik sebelum retry
            
            # Invoke endpoint
            response = runtime_client.invoke_endpoint(
                EndpointName=ENDPOINT_NAME,
                ContentType="application/x-image",
                Body=optimized_image,
                Accept="application/json"
            )
            
            # Parse response
            result = json.loads(response["Body"].read().decode("utf-8"))
            return result
            
        except runtime_client.exceptions.ModelError as e:
            if "timed out" in str(e) and attempt < max_retries - 1:
                st.warning(f"⚠️ Timeout attempt {attempt + 1}. Retrying...")
                continue
            else:
                raise e
        except Exception as e:
            raise e
    
    raise Exception("Max retries exceeded")

# ======================================================
# INISIALISASI AWS CLIENTS
# ======================================================
@st.cache_resource
def get_aws_clients():
    """Initialize AWS clients dengan timeout yang lebih panjang"""
    try:
        # KONFIGURASI PENTING: Tambah timeout
        config = Config(
            read_timeout=300,  # 5 menit (default 60)
            connect_timeout=120,
            retries={
                'max_attempts': 3,
                'mode': 'standard'
            }
        )
        
        # Gunakan IAM Role jika di EC2/SageMaker, atau credentials
        if all([os.getenv("AWS_ACCESS_KEY_ID"), 
                os.getenv("AWS_SECRET_ACCESS_KEY")]):
            session = boto3.Session(
                aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
                region_name=REGION
            )
        else:
            session = boto3.Session(region_name=REGION)
        
        clients = {
            "sagemaker": session.client("sagemaker", config=config),
            "sagemaker_runtime": session.client("sagemaker-runtime", config=config),
            "session": session
        }
        
        return clients
        
    except Exception as e:
        st.error(f"❌ Failed to initialize AWS clients: {str(e)}")
        return None

# ======================================================
# MAIN APPLICATION
# ======================================================
clients = get_aws_clients()

st.title("🧴 Kulit.ai - Optimized Skin Analysis")
st.markdown("**Professional skin type analysis with optimized inference**")

# Sidebar dengan status
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/skin-care.png", width=80)
    st.title("Kulit.ai")
    
    if clients:
        try:
            endpoint_info = clients["sagemaker"].describe_endpoint(
                EndpointName=ENDPOINT_NAME
            )
            status = endpoint_info['EndpointStatus']
            
            if status == 'InService':
                st.success("✅ Endpoint Ready")
                st.caption(f"Status: {status}")
                
                # Show endpoint config
                config_name = endpoint_info['EndpointConfigName']
                st.caption(f"Config: {config_name}")
            else:
                st.warning(f"⚠️ Endpoint: {status}")
        except Exception as e:
            st.error(f"❌ Cannot check endpoint: {str(e)}")
    
    # Mode selection
    st.markdown("---")
    st.markdown("### ⚙️ Inference Mode")
    
    inference_mode = st.radio(
        "Select inference method:",
        ["Direct (Fast)", "Lambda Proxy (Stable)"],
        index=0  # Default to Direct
    )
    
    st.info("""
    **Mode Explanation:**
    - **Direct:** Faster but may timeout
    - **Lambda Proxy:** More stable with retry logic
    """)

# Main interface
uploaded_file = st.file_uploader(
    "Upload face photo (JPG, PNG, JPEG)",
    type=["jpg", "jpeg", "png"],
    help="For best results: close-up, good lighting, no makeup"
)

if uploaded_file:
    if clients:
        # Display image
        col1, col2 = st.columns(2)
        
        with col1:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Show image info
            file_size = len(uploaded_file.getvalue())
            st.caption(f"Original: {file_size/1024:.1f} KB")
            st.caption(f"Dimensions: {image.size[0]}x{image.size[1]}")
        
        with col2:
            st.markdown("### 🎯 Ready to Analyze")
            
            if st.button("🔍 Analyze Skin", type="primary", use_container_width=True):
                with st.spinner("🔄 Processing..."):
                    try:
                        # Read image
                        uploaded_file.seek(0)
                        image_bytes = uploaded_file.read()
                        
                        # Show optimization progress
                        progress_bar = st.progress(0)
                        
                        # Step 1: Optimize image
                        progress_bar.progress(20)
                        st.info("📦 Optimizing image...")
                        
                        optimized_bytes = optimize_image_for_inference(image_bytes)
                        optimized_size = len(optimized_bytes)
                        
                        progress_bar.progress(40)
                        st.success(f"✅ Optimized to: {optimized_size/1024:.1f} KB")
                        
                        # Step 2: Inference based on selected mode
                        progress_bar.progress(60)
                        
                        if inference_mode == "Lambda Proxy" and LAMBDA_FUNCTION_NAME:
                            st.info("🚀 Calling via Lambda proxy...")
                            # For Lambda mode, you'd need to implement this
                            # result = invoke_via_lambda(clients["lambda"], image_bytes)
                            st.warning("Lambda proxy mode not fully implemented yet")
                            progress_bar.progress(100)
                            # Mock result for now
                            result = {
                                "predicted_class": "Normal",
                                "confidence": 0.85,
                                "probabilities": {"Dry": 0.1, "Normal": 0.85, "Oily": 0.05}
                            }
                        else:
                            st.info("🚀 Calling SageMaker endpoint directly...")
                            result = invoke_sagemaker_with_retry(
                                clients["sagemaker_runtime"], 
                                image_bytes,
                                max_retries=2
                            )
                        
                        progress_bar.progress(100)
                        
                        # Display results
                        display_results(result)
                        
                    except Exception as e:
                        st.error(f"❌ Analysis failed: {str(e)}")
                        
                        # Troubleshooting tips
                        with st.expander("🔧 Troubleshooting Tips"):
                            st.markdown("""
                            **Common Solutions:**
                            1. **Reduce image size** before uploading
                            2. **Use Lambda Proxy mode** if available
                            3. **Check CloudWatch logs:** [Link](https://us-east-1.console.aws.amazon.com/cloudwatch/home?region=us-east-1#logsV2:log-groups/log-group/$252Faws$252Fsagemaker$252FEndpoints$252Fkulitai-gpu-prod)
                            4. **Wait 1-2 minutes** and try again (cold start)
                            5. **Contact administrator** to check endpoint health
                            """)
    else:
        st.error("❌ AWS clients not initialized. Check your credentials.")

# Add footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 14px;">
    <p>Kulit.ai v2.0 | AWS SageMaker Endpoint | EfficientNetV2-L</p>
    <p>© 2024 Kulit.ai. All rights reserved.</p>
</div>
""", unsafe_allow_html=True)