import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import csv

from torch.utils.data import random_split

# Create dataset class to read the CSV and loads the images

class RatingsDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.samples= []
        self.transform = transform

        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                self.samples.append((row[0], int(row[1])))

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, rating = self.samples[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)
        
        # Convert rating 1-10 to 0-9 for classification
        label = rating - 1
        return image, label



# Load pretrained ResNet18
model = models.resnet18(weights='DEFAULT')

# Freeze early layers (keep pretrained features)
for param in model.parameters():
    param.requires_grad = False

# Replace final layer (1000 ImageNet classes -> 10 rating classes)
model.fc = nn.Linear(model.fc.in_features, 10)


# Image transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create dataset and dataloader 
dataset = RatingsDataset("ratings.csv", transform=transform)

# Train and Validation split

train_size = int(0.95 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])


train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)



# Training loop
num_epochs = 10

model.train() # Set model to training mode

for epoch in range(num_epochs):
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optmize
        loss.backward()
        optimizer.step()

        # Track stats
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    # Print epoch stats
    epoch_loss = running_loss / len(train_loader)
    accuracy = 100 * correct / total
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%")

    # Validation stats
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
    
    val_accuracy = 100 * val_correct / val_total
    print(f"Validation Loss: {val_loss/len(val_loader):.4f}, Accuracy {val_accuracy:.2f}%")
    
    model.train() # Back to training mode for next epoch

    # Save the model
    torch.save(model.state_dict(), "attractiveness_model.pth")
    print("Model saved!")