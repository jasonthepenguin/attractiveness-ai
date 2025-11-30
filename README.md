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
- [ ] Build labeling tool
- [ ] Label images
- [ ] Train model

## Model

Fine-tuning ResNet18 (pretrained on ImageNet) with a 10-class output layer. Using transfer learning since 1000 images is too small to train from scratch.

## Hosting

Planning to host the model on Hugging Face Spaces (free tier) and build a simple frontend that calls the inference API.

## Dataset

Using CelebA (Celebrity Faces) - 1000 aligned face images stored in `img_align_celeba/`.
