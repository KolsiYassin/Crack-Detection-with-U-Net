from pathlib import Path

from config import DATASET_ROOT


OMNICRACK_ROOT = (
    Path(DATASET_ROOT)
    / "omnicrack30"
    / "train"
)

IMAGE_DIR = OMNICRACK_ROOT / "images"
MASK_DIR = OMNICRACK_ROOT / "masks"

SUBSET_DIR = Path(__file__).parent / "subsets"


def load_omnicrack_subset(size):

    subset_file = SUBSET_DIR / f"{size}.txt"

    if not subset_file.exists():
        raise FileNotFoundError(
            f"Subset file not found: {subset_file}"
        )

    filenames = subset_file.read_text(
        encoding="utf-8"
    ).splitlines()

    samples = []

    for filename in filenames:

        image_path = IMAGE_DIR / filename
        mask_path = MASK_DIR / filename

        if not image_path.exists():
            raise FileNotFoundError(image_path)

        if not mask_path.exists():
            raise FileNotFoundError(mask_path)

        samples.append(
            (image_path, mask_path)
        )

    return samples