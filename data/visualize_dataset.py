from pathlib import Path
import random

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


DATASET_ROOT = Path("F:/Downloads/Damage_detection/Damage_detection/Brix_dataset")

SPLITS = ["train", "val", "test"]
NUM_SAMPLES = 12

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def get_pairs(split):
    image_dir = DATASET_ROOT / split / "images"
    mask_dir = DATASET_ROOT / split / "masks"

    images = {
        p.stem: p
        for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    }

    masks = {
        p.stem: p
        for p in mask_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    }

    common = sorted(set(images) & set(masks))

    return [(images[name], masks[name]) for name in common]


def load_pair(image_path, mask_path):
    image = np.array(Image.open(image_path).convert("RGB"))
    mask = np.array(Image.open(mask_path).convert("L"))

    return image, mask


def create_overlay(image, mask):
    """
    Creates a simple overlay where crack pixels are highlighted.
    """

    overlay = image.copy()

    crack = mask > 0

    # Highlight crack pixels.
    overlay[crack] = [255, 0, 0]

    # Blend original image and highlighted mask.
    result = (0.65 * image + 0.35 * overlay).astype(np.uint8)

    return result


def visualize_samples(split, samples):
    fig, axes = plt.subplots(
        len(samples),
        3,
        figsize=(12, 4 * len(samples))
    )

    if len(samples) == 1:
        axes = np.expand_dims(axes, axis=0)

    fig.suptitle(
        f"{split.upper()} - Crack Segmentation Samples",
        fontsize=16
    )

    for row, (image_path, mask_path) in enumerate(samples):

        image, mask = load_pair(image_path, mask_path)
        overlay = create_overlay(image, mask)

        axes[row, 0].imshow(image)
        axes[row, 0].axis("off")

        axes[row, 1].imshow(mask, cmap="gray", vmin=0, vmax=255)
        axes[row, 1].axis("off")

        axes[row, 2].imshow(overlay)
        axes[row, 2].axis("off")

    plt.tight_layout()
    plt.show()


def main():

    print("Loading dataset...")

    all_pairs = {}

    for split in SPLITS:
        pairs = get_pairs(split)
        all_pairs[split] = pairs

        print(f"{split}: {len(pairs)} pairs")

    # Random samples from each split
    for split in SPLITS:

        pairs = all_pairs[split]

        num_samples = min(NUM_SAMPLES, len(pairs))

        samples = random.sample(pairs, num_samples)

        visualize_samples(split, samples)

    # Specifically visualize empty masks
    print("\nSearching for empty masks...")

    empty_samples = []

    for image_path, mask_path in all_pairs["train"]:

        mask = np.array(
            Image.open(mask_path).convert("L")
        )

        if np.sum(mask > 0) == 0:
            empty_samples.append((image_path, mask_path))

    print(f"Found {len(empty_samples)} empty masks.")

    if empty_samples:

        samples = random.sample(
            empty_samples,
            min(6, len(empty_samples))
        )

        visualize_samples(
            "TRAIN - EMPTY MASKS",
            samples
        )


if __name__ == "__main__":
    main()