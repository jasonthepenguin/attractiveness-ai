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

        # Return rating as float for regression
        label = float(rating)
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

# Freeze early layers, unfreeze layer3 and layer4 for fine-tuning
for name, param in model.named_parameters():
    if "layer4" in name or "layer3" in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

# Replace final layer with single output for regression
model.fc = nn.Linear(model.fc.in_features, 1)
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

# Train and Validation split (85/15 for more reliable validation)
train_size = int(0.85 * len(dataset))
val_size = len(dataset) - train_size
train_subset, val_subset = random_split(dataset, [train_size, val_size])

# Wrap with transforms
train_dataset = TransformDataset(train_subset, train_transform)
val_dataset = TransformDataset(val_subset, val_transform)


train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Loss function and optimizer (train all unfrozen parameters)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

# Training loop
num_epochs = 50
early_stop_patience = 7
epochs_without_improvement = 0

best_val_loss = float('inf')

model.train() # Set model to training mode

for epoch in range(num_epochs):
    running_loss = 0.0
    running_mae = 0.0
    total = 0

    for images, labels in train_loader:

        # Move data to device, reshape labels for regression
        images = images.to(device)
        labels = labels.float().unsqueeze(1).to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Track stats
        running_loss += loss.item()
        running_mae += torch.abs(outputs - labels).sum().item()
        total += labels.size(0)

    # Print epoch stats
    epoch_loss = running_loss / len(train_loader)
    epoch_mae = running_mae / total
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, MAE: {epoch_mae:.2f}")

    # Validation stats
    model.eval()
    val_loss = 0.0
    val_mae = 0.0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:

            # Move data to device, reshape labels for regression
            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            val_mae += torch.abs(outputs - labels).sum().item()
            val_total += labels.size(0)

    avg_val_loss = val_loss / len(val_loader)
    avg_val_mae = val_mae / val_total
    print(f"Validation Loss: {avg_val_loss:.4f}, MAE: {avg_val_mae:.2f}")

    # Step the scheduler
    scheduler.step(avg_val_loss)

    model.train() # Back to training mode for next epoch

    # Save the model if validation loss improved
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        epochs_without_improvement = 0
        torch.save(model.state_dict(), "attractiveness_model.pth")
        print(f"New best model saved! Val Loss: {avg_val_loss:.4f}, MAE: {avg_val_mae:.2f}")
    else:
        epochs_without_improvement += 1
        if epochs_without_improvement >= early_stop_patience:
            print(f"Early stopping after {epoch + 1} epochs")
            break

print(f"Training complete! Best validation loss: {best_val_loss:.4f}")