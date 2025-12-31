import cv2
import numpy as np
import torch
from torchvision import transforms

def preprocess_image(image_bytes):
    """
    Fungsi ini mengubah bytes gambar (dari upload user)
    menjadi Tensor yang siap dimakan oleh EfficientNet.
    """
    # 1. Decode bytes ke OpenCV Image
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # 2. Convert BGR (OpenCV) ke RGB (PyTorch standard)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 3. Resize & Normalize (Sesuai standar ImageNet/EfficientNet)
    transform_pipeline = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img_tensor = transform_pipeline(img)
    
    # 4. Tambah dimensi batch (1, C, H, W) karena model butuh batch
    return img_tensor.unsqueeze(0)