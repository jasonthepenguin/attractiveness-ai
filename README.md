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

## Dataset

Using CelebA (Celebrity Faces) - 1000 aligned face images stored in `img_align_celeba/`.
