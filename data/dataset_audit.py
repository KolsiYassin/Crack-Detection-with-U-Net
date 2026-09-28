from pathlib import Path
from collections import Counter
from PIL import Image
import numpy as np


DATASET_ROOT = Path("F:\Downloads\Damage_detection\Damage_detection\Brix_dataset")

SPLITS = ["train", "val", "test"]


def get_files(directory):
    """Return image files supported by PIL."""
    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    return {
        p.stem: p
        for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in extensions
    }


def analyze_mask(mask_path):
    """Analyze a segmentation mask."""
    mask = np.array(Image.open(mask_path))

    unique_values, counts = np.unique(mask, return_counts=True)

    return {
        "shape": mask.shape,
        "unique_values": unique_values.tolist(),
        "num_pixels": mask.size,
        "crack_pixels": int((mask == 255).sum()),
        "crack_ratio": float((mask == 255).mean()),
        "unique_counts": dict(zip(unique_values.tolist(), counts.tolist())),
    }


def analyze_split(split):
    print("\n" + "=" * 70)
    print(f" {split.upper()}")
    print("=" * 70)

    image_dir = DATASET_ROOT / split / "images"
    mask_dir = DATASET_ROOT / split / "masks"

    if not image_dir.exists():
        print(f"ERROR: Missing directory: {image_dir}")
        return

    if not mask_dir.exists():
        print(f"ERROR: Missing directory: {mask_dir}")
        return

    images = get_files(image_dir)
    masks = get_files(mask_dir)

    print(f"Images: {len(images)}")
    print(f"Masks:  {len(masks)}")

    missing_masks = sorted(set(images) - set(masks))
    missing_images = sorted(set(masks) - set(images))

    if missing_masks:
        print(f"\nImages without masks: {len(missing_masks)}")
        print(missing_masks[:10])

    if missing_images:
        print(f"\nMasks without images: {len(missing_images)}")
        print(missing_images[:10])

    if missing_masks or missing_images:
        print("\nWARNING: Image/mask mismatch!")
    else:
        print("\nImage/mask pairing: OK")

    resolutions = Counter()
    mask_values = Counter()
    empty_masks = 0
    crack_ratios = []

    dimension_mismatches = []

    for stem in sorted(set(images) & set(masks)):
        image_path = images[stem]
        mask_path = masks[stem]

        image = Image.open(image_path)
        image_size = image.size  # (width, height)

        analysis = analyze_mask(mask_path)

        mask_shape = analysis["shape"]

        # Mask should have height x width
        if len(mask_shape) >= 2:
            mask_size = (mask_shape[1], mask_shape[0])

            if image_size != mask_size:
                dimension_mismatches.append(
                    (stem, image_size, mask_size)
                )

        resolutions[image_size] += 1

        for value in analysis["unique_values"]:
            mask_values[value] += 1

        if analysis["crack_pixels"] == 0:
            empty_masks += 1

        crack_ratios.append(analysis["crack_ratio"])

    print("\nImage resolutions:")

    for resolution, count in resolutions.most_common():
        print(f"  {resolution}: {count}")

    print("\nMask values encountered:")

    for value, count in sorted(mask_values.items()):
        print(f"  {value}: {count} masks")

    print(f"\nEmpty masks: {empty_masks}")

    if crack_ratios:
        print("\nCrack pixel ratio:")
        print(f"  Minimum: {min(crack_ratios):.6f}")
        print(f"  Maximum: {max(crack_ratios):.6f}")
        print(f"  Mean:    {np.mean(crack_ratios):.6f}")
        print(f"  Median:  {np.median(crack_ratios):.6f}")

    if dimension_mismatches:
        print("\nDimension mismatches:")
        for item in dimension_mismatches[:10]:
            print(f"  {item}")
    else:
        print("\nImage/mask dimensions: OK")


def main():
    print("=" * 70)
    print("CRACK SEGMENTATION DATASET AUDIT")
    print("=" * 70)

    print(f"\nDataset: {DATASET_ROOT.resolve()}")

    for split in SPLITS:
        analyze_split(split)

    print("\n" + "=" * 70)
    print("AUDIT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()