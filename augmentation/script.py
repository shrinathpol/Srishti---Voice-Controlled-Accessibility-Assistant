import cv2
import albumentations as A
import os
import glob

# --- Configuration ---
# 1. Path to the folder with your source images
INPUT_DIR = r"E:\Program\phase-1 project\EXPERIMENTAL-1\project\set" # <--- Change this to your folder name

# 2. Directory where augmented images will be saved
OUTPUT_DIR = r"E:\Program\phase-1 project\EXPERIMENTAL-1\augmentation\saved"

# 3. Number of augmented versions to generate PER original image
AUGMENTATIONS_PER_IMAGE = 50

# --- End of Configuration ---


# Find all image files in the input directory
image_paths = glob.glob(os.path.join(INPUT_DIR, '*.jpg')) + \
              glob.glob(os.path.join(INPUT_DIR, '*.jpeg')) + \
              glob.glob(os.path.join(INPUT_DIR, '*.png'))

if not image_paths:
    print(f"Error: No images found in the '{INPUT_DIR}' folder.")
    print("Please create the folder and add images (e.g., .jpg, .png) to it.")
    exit()

# Create the main output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)


# Define a strong augmentation pipeline using Albumentations
# Each transform has a probability 'p' of being applied.
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=30, p=0.7, border_mode=cv2.BORDER_CONSTANT),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=25, p=0.8, border_mode=cv2.BORDER_CONSTANT),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.6),
    A.GaussNoise(var_limit=(10.0, 40.0), p=0.5),
    A.GaussianBlur(blur_limit=(3, 7), p=0.4),
    A.GridDistortion(p=0.5),
    A.CoarseDropout(max_holes=8, max_height=16, max_width=16, min_holes=1, fill_value=0, p=0.5),
])

print(f"Found {len(image_paths)} images to augment.")
total_generated = 0

# Loop over each image path
for image_path in image_paths:
    try:
        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Could not read image {image_path}. Skipping.")
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Get the original filename without the extension
        original_filename = os.path.splitext(os.path.basename(image_path))[0]

        # Create a specific sub-folder for this image's augmentations
        image_output_dir = os.path.join(OUTPUT_DIR, original_filename)
        os.makedirs(image_output_dir, exist_ok=True)

        print(f"-> Processing {original_filename}...")

        # Generate and save the augmented images
        for i in range(AUGMENTATIONS_PER_IMAGE):
            augmented = transform(image=image)
            augmented_image = augmented['image']
            augmented_image_bgr = cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR)

            # Define the output filename
            output_filename = os.path.join(image_output_dir, f"{original_filename}_aug_{i+1}.jpg")
            cv2.imwrite(output_filename, augmented_image_bgr)
            total_generated += 1

    except Exception as e:
        print(f"Error processing {image_path}: {e}")

print("\n" + "="*30)
print(f"✅ Augmentation Complete!")
print(f"Total images generated: {total_generated}")
print(f"Results saved in the '{OUTPUT_DIR}' folder.")
print("="*30)