import torch
from torchvision import datasets, transforms, models
import torch.nn as nn
import os

# --- KONFIGURASI ---
DATA_DIR = 'data/test'
MODEL_PATH = 'notebooks/model_hasil_training/model.pth'
BATCH_SIZE = 8
IMG_SIZE = 384  # HARUS SAMA DENGAN TRAINING

def evaluate():
    print("🚀 Memulai Evaluasi Final Model AWS")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"⚙️ Device: {device}")

    # ---------------- TRANSFORM (NO AUGMENTATION) ----------------
    test_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    if not os.path.exists(DATA_DIR):
        print(f"❌ Folder {DATA_DIR} tidak ditemukan")
        return

    test_dataset = datasets.ImageFolder(DATA_DIR, transform=test_transforms)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False
    )

    print(f"📂 Total Data Test: {len(test_dataset)}")
    print(f"🏷️ Classes: {test_dataset.classes}")

    # ---------------- LOAD MODEL ----------------
    model = models.efficientnet_v2_l(weights=None)
    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        len(test_dataset.classes)
    )

    if not os.path.exists(MODEL_PATH):
        print("❌ model.pth tidak ditemukan")
        return

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print("✅ Model AWS berhasil dimuat")

    model = model.to(device)
    model.eval()

    # ---------------- EVALUATION ----------------
    correct = 0
    total = 0

    print("\n📝 Evaluasi berjalan...")

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (preds == labels).sum().item()

    accuracy = 100 * correct / total
    print("-" * 40)
    print(f"🏆 FINAL TEST ACCURACY: {accuracy:.2f}%")
    print("-" * 40)

if __name__ == "__main__":
    evaluate()
