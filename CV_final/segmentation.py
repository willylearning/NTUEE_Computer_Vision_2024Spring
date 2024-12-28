from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
from PIL import Image
import torch
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import cm
import os
import json
import numpy as np
from scipy.ndimage import find_objects, label
from skimage.measure import find_contours
import cv2

# Load the processor and model
processor = AutoImageProcessor.from_pretrained(
    "facebook/mask2former-swin-base-coco-panoptic")
model = Mask2FormerForUniversalSegmentation.from_pretrained(
    "facebook/mask2former-swin-base-coco-panoptic")


def draw_and_save_segmentation_with_contours(segmentation, segments_info, save_dir):
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)

    instances_counter = defaultdict(int)
    for segment in segments_info:
        label_id = segment['label_id']
        fig, ax = plt.subplots()
        segment_id = segment['id']
        segment_label_id = segment['label_id']
        segment_label = model.config.id2label[segment_label_id]
        label = f"{segment_label}-{instances_counter[segment_label_id]}"
        instances_counter[segment_label_id] += 1

        mask = segmentation == segment_id

        # Find contours
        contours = find_contours(mask.numpy(), level=0.5)

        # Plot and save the segmentation mask
        ax.imshow(mask, cmap='Greys_r')
        ax.axis('off')
        plt.tight_layout()
        filename = os.path.join(save_dir, f"segment_{label}.png")
        plt.savefig(filename, bbox_inches='tight',
                    pad_inches=0, transparent=True)
        plt.close()


# CHECK the PATH!
# Directory paths
input_dir = "./golden"
output_dir = "./segmentation"

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# List all files in the input directory
all_files = [f for f in os.listdir(input_dir) if f.endswith(
    ".png") or f.endswith(".jpg") or f.endswith(".jpeg")]
result = all_files[0:129]
print(result)

for filename in result:
    if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg"):
        # Load the image
        image_path = os.path.join(input_dir, filename)
        image = Image.open(image_path)

        # Convert grayscale image to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Process the image and perform segmentation
        inputs = processor(images=image, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)

        # Post-process the segmentation output
        prediction = processor.post_process_panoptic_segmentation(
            outputs, target_sizes=[image.size[::-1]])[0]

        # Extract panoptic segmentation results
        segmentation = prediction['segmentation']
        segments_info = prediction['segments_info']

        # Save the segmentation with contours
        save_path = os.path.join(output_dir, filename)
        draw_and_save_segmentation_with_contours(
            segmentation, segments_info, save_path)
