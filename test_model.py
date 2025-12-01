
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog

# Load model (same architecture as training)
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 10)
model.load_state_dict(torch.load("attractiveness_model.pth", map_location="cpu"))
model.eval()

# Same transform as validation (no augmentation)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict(image_path):
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0) # Add batch dimension

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)[0] # Get probabilities

        # Weighted average: sum(rating * probability) for all ratings
        ratings = torch.arange(1, 11, dtype=torch.float) # [1, 2, 3.... 10]
        weighted_score = (probabilities * ratings).sum().item()


        #predicted_class = torch.argmax(output, dim=1).item()
    
    #return predicted_class + 1 # Convert 0-9 back to 1-10
    return weighted_score


# GUI
root = tk.Tk()
root.title("Attractiveness Predictor")
root.geometry("400x500")

image_label = tk.Label(root)
image_label.pack(pady=20)

result_label = tk.Label(root, text="Select an image", font=("Arial", 24))
result_label.pack(pady=20)

def select_image():
    path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if path:
        # Display Image
        img = Image.open(path)
        img.thumbnail((300, 300))
        photo = ImageTk.PhotoImage(img)
        image_label.config(image=photo)
        image_label.image = photo

        # Predict and show result
        rating = predict(path)
        #result_label.config(text=f"Rating: {rating}/10")
        result_label.config(text=f"Rating: {rating:.2f}/10")

btn = tk.Button(root, text="Select Image", command=select_image, font=("Arial", 14))
btn.pack(pady=10)

root.mainloop()