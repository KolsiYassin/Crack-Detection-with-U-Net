from pathlib import Path
from config import DATASET_ROOT


ROOT = Path(DATASET_ROOT) / "omnicrack30" / "train"



IMAGE_DIR = ROOT / "images"
MASK_DIR = ROOT / "masks"


def main():

    images = {
        path.stem
        for path in IMAGE_DIR.iterdir()
        if path.is_file()
    }

    masks = {
        path.stem
        for path in MASK_DIR.iterdir()
        if path.is_file()
    }

    print("OmniCrack30k training set")
    print("-------------------------")

    print(f"Images: {len(images)}")
    print(f"Masks:  {len(masks)}")

    missing_masks = images - masks
    missing_images = masks - images

    print()
    print(f"Images without masks: {len(missing_masks)}")
    print(f"Masks without images: {len(missing_images)}")

    if missing_masks:
        print("\nExample missing masks:")
        for name in sorted(missing_masks)[:10]:
            print(name)

    if missing_images:
        print("\nExample missing images:")
        for name in sorted(missing_images)[:10]:
            print(name)


if __name__ == "__main__":
    main()