import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import csv

from torch.utils.data import random_split

torch.manual_seed(42)

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


        
        # Convert rating 1-10 to 0-9 for classification
        label = rating - 1
        return image, label

class TransformDataset:
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        image, label = self.subset[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# Load pretrained ResNet18
model = models.resnet18(weights='DEFAULT')

# Use Metal to use Macs GPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Freeze early layers (keep pretrained features)
for param in model.parameters():
    param.requires_grad = False

# Replace final layer (1000 ImageNet classes -> 10 rating classes)
model.fc = nn.Linear(model.fc.in_features, 10)
model = model.to(device)

# Image transforms
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create dataset and dataloader 
dataset = RatingsDataset("ratings.csv", transform=None)

# Train and Validation split
train_size = int(0.95 * len(dataset))
val_size = len(dataset) - train_size
#train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_subset, val_subset = random_split(dataset, [train_size, val_size])

# Wrap with transforms
train_dataset = TransformDataset(train_subset, train_transform)
val_dataset = TransformDataset(val_subset, val_transform)


train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)



# Training loop
num_epochs = 10

best_val_accuracy = 0.0

model.train() # Set model to training mode

for epoch in range(num_epochs):
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        # Move data from CPU to the GPU
        images, labels = images.to(device), labels.to(device)

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

            # Move data from CPU to the GPU
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
    
    val_accuracy = 100 * val_correct / val_total
    print(f"Validation Loss: {val_loss/len(val_loader):.4f}, Accuracy {val_accuracy:.2f}%")
    
    model.train() # Back to training mode for next epoch


    # Save the model if validaiton accuracy is better than last epoch
    if (val_accuracy > best_val_accuracy):
        best_val_accuracy = val_accuracy
        torch.save(model.state_dict(), "attractiveness_model.pth")
        print(f"New best model saved! Accuracy : {val_accuracy:.2f}%")

print(f"Training complete! Best validation accuracy is : {best_val_accuracy:.2f}%")