import torch
import torch.nn as nn
from torchvision import models, transforms
import os
import json
import cv2
import numpy as np

# ======================================================
# GLOBAL CONFIG
# ======================================================
MAX_IMAGE_SIZE = 5 * 1024 * 1024  # 5 MB
IMG_SIZE = 224
CLASS_NAMES = ["Dry", "Normal", "Oily"]

# ======================================================
# 1. MODEL LOADER
# ======================================================
def model_fn(model_dir):
    print("🔥 [model_fn] Loading model...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔥 Using device: {device}")

    model = models.efficientnet_v2_s(weights=None)
    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features, len(CLASS_NAMES)
    )

    model_path = os.path.join(model_dir, "model.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ model.pth not found in {model_dir}")

    model.load_state_dict(
        torch.load(model_path, map_location=device)
    )

    model.to(device)
    model.eval()
    print("✅ Model loaded & ready")
    return model

# ======================================================
# 2. INPUT HANDLER
# ======================================================
def input_fn(request_body, request_content_type):
    print("🔥 [input_fn] Called")
    try:
        if request_body is None or len(request_body) == 0:
            raise ValueError("❌ Empty request body")

        if len(request_body) > MAX_IMAGE_SIZE:
            raise ValueError("❌ Image too large (>5MB)")

        if request_content_type not in (
            "application/octet-stream",
            "application/x-image",
            "image/jpeg",
            "image/png",
        ):
            raise ValueError(f"❌ Unsupported content type: {request_content_type}")

        # Decode image
        nparr = np.frombuffer(request_body, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("❌ Failed to decode image")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Transform
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        tensor = transform(img).unsqueeze(0)
        print("✅ Image preprocessed:", tensor.shape)
        return tensor

    except Exception as e:
        print(f"❌ Error in input_fn: {e}")
        raise e

# ======================================================
# 3. PREDICT FUNCTION
# ======================================================
def predict_fn(input_object, model):
    print("🔥 [predict_fn] Running inference")
    try:
        device = next(model.parameters()).device
        input_object = input_object.to(device)

        with torch.no_grad():
            # Optional: FP16 untuk GPU lebih cepat
            if device.type == "cuda":
                input_object = input_object.half()
                model.half()

            outputs = model(input_object)

            # Sinkronisasi GPU
            if device.type == "cuda":
                torch.cuda.synchronize()

        print("✅ Inference done")
        return outputs

    except Exception as e:
        print(f"❌ Error in predict_fn: {e}")
        raise e

# ======================================================
# 4. OUTPUT HANDLER
# ======================================================
def output_fn(predictions, accept):
    print("🔥 [output_fn] Formatting output")
    try:
        probs = torch.softmax(predictions, dim=1)
        conf, idx = torch.max(probs, 1)

        result = {
            "prediction": CLASS_NAMES[idx.item()],
            "confidence": round(conf.item() * 100, 2),
            "probabilities": {
                CLASS_NAMES[i]: round(probs[0][i].item() * 100, 2)
                for i in range(len(CLASS_NAMES))
            },
        }

        response = json.dumps(result)
        if accept == "application/json":
            return response, accept
        return response, "application/json"

    except Exception as e:
        print(f"❌ Error in output_fn: {e}")
        raise e
