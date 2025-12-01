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
- [x] Deploy to Hugging Face Spaces
- [ ] Build Next.js frontend

## Model

Fine-tuning ResNet18 (pretrained on ImageNet) with regression output (single value). Using transfer learning since 1000 images is too small to train from scratch.

## Hugging Face Space

- **Space URL:** https://huggingface.co/spaces/jasonfor2020/jb-hot-regression
- **Gradio App:** https://jasonfor2020-jb-hot-regression.hf.space

### API Usage (for Next.js)

Using `@gradio/client`:
```typescript
import { Client } from "@gradio/client";

const client = await Client.connect("jasonfor2020/jb-hot-regression");
const result = await client.predict("/predict", { image: imageFile });
const rating = result.data[0]; // e.g. "7.2 / 10"
```

Or via fetch:
```typescript
const response = await fetch("https://jasonfor2020-jb-hot-regression.hf.space/api/predict", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ data: [base64Image] })
});
const { data } = await response.json();
const rating = data[0];
```

## Dataset

Using CelebA (Celebrity Faces) - 2000 aligned face images stored in `img_align_celeba/`.
