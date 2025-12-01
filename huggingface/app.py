import gradio as gr
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

# Load model
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 1)
model.load_state_dict(torch.load("attractiveness_model.pth", map_location="cpu"))
model.eval()

# Same transforms as validation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict(image):
    if image is None:
        return "Please upload an image"

    # Convert to PIL if needed
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    image = image.convert("RGB")
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        rating = output.item()

    # Clamp to 1-10 range
    rating = max(1.0, min(10.0, rating))

    return f"{rating:.1f} / 10"

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload a face image"),
    outputs=gr.Text(label="Attractiveness Rating"),
    title="Attractiveness Predictor",
    description="Upload a face image to get an attractiveness rating (1-10)"
)

demo.launch()
