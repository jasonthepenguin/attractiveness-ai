import os
import glob

# Get all images sorted
images = sorted(glob.glob("./new_celeba/*.jpg"))

# Keep only images after 001000 (next 1000 images: 001001 - 00200)
images_to_keep = []

for img in images: 
    # Extract num from filename eg "001001" from "001001.jpg"
    filename = os.path.basename(img)
    num = int(filename.split('.')[0])

    if 1001 <= num <= 2000:
        images_to_keep.append(img)
    

# Delete everything else
for img in images:
    if img not in images_to_keep:
        os.remove(img)

print(f"Kept {len(images_to_keep)} images (001001-002000)")
print(f"Deleted {len(images) - len(images_to_keep)} images")