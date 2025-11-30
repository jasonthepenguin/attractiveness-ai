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