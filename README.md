# Attractiveness AI

Training a personal face attractiveness rating model (1-10 scale) based on my own preferences.

## Approach

1. Get a dataset of face images
2. Manually label each image with a 1-10 rating
3. Train a small model on the labeled data
4. Use the model to predict attractiveness scores on new faces

## Current Progress

- [x] Downloaded CelebA dataset
- [x] Trimmed to 1000 images for labeling
- [x] Build labeling tool
- [x] Label images
- [x] Train model
- [ ] Deploy to Hugging Face Spaces
- [ ] Build Next.js frontend

## Model

Fine-tuning ResNet18 (pretrained on ImageNet) with regression output (single value). Using transfer learning since 1000 images is too small to train from scratch.

## Deployment Plan

### Architecture
```
Next.js App  →  Hugging Face Spaces API  →  Returns rating
(frontend)         (hosts model)
```

### Step 1: Host model on Hugging Face Spaces
- Create a Gradio app that loads `attractiveness_model.pth`
- Upload to HF Spaces (free tier)
- This auto-generates an API endpoint

### Step 2: Build Next.js frontend
- Simple image upload UI
- Call HF Spaces API using `@gradio/client` or fetch
- Display the rating result

## Dataset

Using CelebA (Celebrity Faces) - 1000 aligned face images stored in `img_align_celeba/`.
