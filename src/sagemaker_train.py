import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import copy
from sklearn.metrics import classification_report

def train():
    print("🚀 Starting Training EfficientNetV2-L")

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.00001)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--patience', type=int, default=3)

    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST'))
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR', './model'))

    args = parser.parse_args()
    os.makedirs(args.model_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"⚙️ Using device: {device}")

    # ---------------- DATA ----------------
    IMG_SIZE = 384

    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(args.train, transform=train_transform)
    test_dataset = datasets.ImageFolder(args.test, transform=val_transform)
    class_names = train_dataset.classes

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
    )

    print(f"✅ Classes: {class_names}")

    # ---------------- MODEL ----------------
    weights = models.EfficientNet_V2_L_Weights.DEFAULT
    model = models.efficientnet_v2_l(weights=weights)

    # Freeze backbone
    for param in model.features.parameters():
        param.requires_grad = True

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, len(class_names))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler()

    # ---------------- EARLY STOPPING ----------------
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_wts = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_preds, best_labels = None, None

    # ---------------- TRAIN LOOP ----------------
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * inputs.size(0)

        train_loss /= len(train_dataset)

        # -------- VALIDATION --------
        model.eval()
        val_loss = 0.0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_loss /= len(test_dataset)

        # ✅ PRINT TIAP EPOCH
        print(f"Epoch [{epoch+1}/{args.epochs}] "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # -------- EARLY STOP CHECK --------
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            best_preds = all_preds
            best_labels = all_labels
            patience_counter = 0
            print("✅ Model membaik (best updated)")
        else:
            patience_counter += 1
            print(f"⚠️ No improvement ({patience_counter}/{args.patience})")

            if patience_counter >= args.patience:
                print("🛑 Early Stopping Triggered")
                break

    # ---------------- FINAL REPORT ----------------
    print("\n📊 CLASSIFICATION REPORT (BEST / LAST EPOCH)")
    print(f"🏆 Best Epoch: {best_epoch}")
    print(classification_report(best_labels, best_preds, target_names=class_names))

    # ---------------- SAVE MODEL ----------------
    model.load_state_dict(best_model_wts)
    model_path = os.path.join(args.model_dir, "model.pth")
    torch.save(model.state_dict(), model_path)

    print(f"💾 Model disimpan di: {model_path}")
    print("✅ Training selesai")

if __name__ == "__main__":
    train()
