from pathlib import Path
import random

from config import DATASET_ROOT


ROOT = Path(DATASET_ROOT) / "omnicrack30" / "train"
IMAGE_DIR = ROOT / "images"

OUTPUT_DIR = Path(__file__).parent / "subsets"

SEED = 42


def main():

    images = sorted(
        path.name
        for path in IMAGE_DIR.iterdir()
        if path.is_file()
    )

    print(f"Available OmniCrack images: {len(images)}")

    if len(images) < 5000:
        raise RuntimeError(
            "Not enough images to create a 5000-image subset."
        )

    rng = random.Random(SEED)

    selected = images.copy()
    rng.shuffle(selected)

    subset_3000 = selected[:3000]
    subset_5000 = selected[:5000]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    path_3000 = OUTPUT_DIR / "3000.txt"
    path_5000 = OUTPUT_DIR / "5000.txt"

    path_3000.write_text(
        "\n".join(subset_3000),
        encoding="utf-8",
    )

    path_5000.write_text(
        "\n".join(subset_5000),
        encoding="utf-8",
    )

    print()
    print("Subsets created")
    print("----------------")
    print(f"3000: {path_3000}")
    print(f"5000: {path_5000}")

    print()
    print("Verification")
    print("------------")
    print(f"3000 subset: {len(subset_3000)}")
    print(f"5000 subset: {len(subset_5000)}")

    overlap = set(subset_3000) & set(subset_5000)

    print(f"Overlap:     {len(overlap)}")


if __name__ == "__main__":
    main()