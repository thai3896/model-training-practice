import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import os

# 1. SETUP THE DATA PIPELINE (The "Eyes" of the AI)
# We need to resize every image to 224x224 pixels so they all match mathematically.
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(), # A little data augmentation!
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # Standard ImageNet colors
])

# Point PyTorch at your dataset folders
# It expects a structure like:
# dataset/
#   pho/ (put all Pho images here)
#   not_pho/ (put Bun Bo, Ramen, Banh Mi here)
data_dir = 'dataset'
if not os.path.exists(os.path.join(data_dir, 'pho')):
    print("⚠️ Please put your images in 'dataset/pho' and 'dataset/not_pho' before running!")
    exit()

image_dataset = datasets.ImageFolder(data_dir, data_transforms)
dataloader = DataLoader(image_dataset, batch_size=32, shuffle=True)
class_names = image_dataset.classes
print(f"Found {len(image_dataset)} total images belonging to classes: {class_names}")

# 2. LOAD THE BASE BRAIN (ResNet-18)
# This model already knows what edges, curves, and textures look like!
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Training on device: {device}")

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# 3. SWAP THE FINAL BUCKET
# ResNet normally has 1000 output buckets (Dog, Cat, Car, etc.)
# We rip off the final layer and replace it with just 2 buckets: "Pho" or "Not Pho"
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2) 
model = model.to(device)

# 4. SET UP THE GYM
criterion = nn.CrossEntropyLoss()
# We only train the new final layer we just attached! (Like LoRA, but for vision)
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

# 5. TRAIN!
num_epochs = 5
print("Starting training...")

for epoch in range(num_epochs):
    print(f'Epoch {epoch+1}/{num_epochs}')
    print('-' * 10)

    model.train()
    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Forward Pass
        optimizer.zero_grad()
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        
        # Calculate Error and Tweak Dials
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(image_dataset)
    epoch_acc = running_corrects.double() / len(image_dataset)
    print(f'Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

# 6. SAVE YOUR NEW MODEL
torch.save(model.state_dict(), 'pho_detector.pth')
print("Training complete! Model saved as pho_detector.pth")
