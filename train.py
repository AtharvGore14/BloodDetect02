import os
import torch
import torch.nn as nn
import torch.optim as optim
<<<<<<< HEAD
from torchvision import datasets, transforms
=======
from torchvision import datasets, transforms, models
from torchvision.models import ResNet18_Weights
from torch.optim.lr_scheduler import StepLR
>>>>>>> 8a39080 (bloodDetect)
from torch.utils.data import DataLoader
from PIL import ImageFile

# Allow loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

<<<<<<< HEAD
# Define Simple CNN Model
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 28 * 28, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x

def main():
    DATA_DIR = 'dataset/dataset_blood_group'
    MODEL_SAVE_PATH = 'fingerprint_blood_group_model.pth'

    # Image transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # Dataset and DataLoader
    dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
    data_loader = DataLoader(dataset, batch_size=16, shuffle=True)

    num_classes = len(dataset.classes)
    print(f"📁 Found {len(dataset)} images across {num_classes} classes: {dataset.classes}")

    # Model, loss, optimizer
    model = SimpleCNN(num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
=======
def main():
    print("🚀 Training started...")

    DATA_DIR = 'dataset/dataset_blood_group'
    MODEL_SAVE_PATH = 'fingerprint_blood_group_model.pth'

    # 🔁 Advanced Data Augmentation
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 📁 Dataset loading
    train_dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    print(f"✅ Found {len(train_dataset)} images across {len(train_dataset.classes)} classes: {train_dataset.classes}")

    # 🧠 Load Pretrained ResNet18 & Unfreeze All Layers
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = True
    model.fc = nn.Linear(model.fc.in_features, len(train_dataset.classes))
>>>>>>> 8a39080 (bloodDetect)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

<<<<<<< HEAD
    # Training loop
=======
    # ✅ Loss Function + Optimizer + Scheduler
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

    # 🔁 Training Loop
>>>>>>> 8a39080 (bloodDetect)
    EPOCHS = 20
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

<<<<<<< HEAD
        for inputs, labels in data_loader:
=======
        for inputs, labels in train_loader:
>>>>>>> 8a39080 (bloodDetect)
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

<<<<<<< HEAD
        epoch_loss = running_loss / len(data_loader)
        epoch_acc = 100. * correct / total
        print(f"📊 Epoch [{epoch + 1}/{EPOCHS}] | Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.2f}%")

    # Save model
=======
        scheduler.step()
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        print(f"📊 Epoch [{epoch+1}/{EPOCHS}] | Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.2f}%")

    # 💾 Save Trained Model
>>>>>>> 8a39080 (bloodDetect)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"✅ Model saved at '{MODEL_SAVE_PATH}'")

if __name__ == "__main__":
    main()
