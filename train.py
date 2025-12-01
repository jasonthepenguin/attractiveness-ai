import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import csv

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
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)