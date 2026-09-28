import os

import torch
from torch.utils.data import DataLoader, Subset

from data.dataset import CrackDataset
from models.unet.unet import UNet
from config import (
    DATASET_ROOT,
    IMAGE_SIZE,
    BATCH_SIZE,
    NUM_WORKERS,
    CHECKPOINT_DIR,
)


def get_source(filename):

    filename = filename.lower()

    if filename.startswith("s2ds_"):
        return "S2DS"

    if filename.startswith("rub_"):
        return "RUB"

    if filename.startswith("crackseg_"):
        return "Crack-Seg"

    return "Unknown"


def evaluate_source(model, dataset, device):

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    all_dice = []
    all_iou = []

    non_empty_dice = []
    non_empty_iou = []

    empty_count = 0

    with torch.no_grad():

        for images, masks in loader:

            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)

            probabilities = torch.sigmoid(logits)

            predictions = (
                probabilities > 0.5
            ).float()

            # Flatten each image separately.
            predictions = predictions.view(
                predictions.size(0), -1
            )

            masks = masks.view(
                masks.size(0), -1
            )

            intersection = (
                predictions * masks
            ).sum(dim=1)

            prediction_area = (
                predictions
            ).sum(dim=1)

            target_area = (
                masks
            ).sum(dim=1)

            dice = (
                2 * intersection + 1e-7
            ) / (
                prediction_area
                + target_area
                + 1e-7
            )

            union = (
                prediction_area
                + target_area
                - intersection
            )

            iou = (
                intersection + 1e-7
            ) / (
                union + 1e-7
            )

            # Store per-image results.
            all_dice.extend(
                dice.cpu().tolist()
            )

            all_iou.extend(
                iou.cpu().tolist()
            )

            # Ground-truth mask contains crack pixels.
            non_empty = target_area > 0

            empty_count += (
                (~non_empty)
                .sum()
                .item()
            )

            non_empty_dice.extend(
                dice[non_empty]
                .cpu()
                .tolist()
            )

            non_empty_iou.extend(
                iou[non_empty]
                .cpu()
                .tolist()
            )

    # Mean over all images.
    mean_dice = (
        sum(all_dice) / len(all_dice)
    )

    mean_iou = (
        sum(all_iou) / len(all_iou)
    )

    # Mean only over images with actual cracks.
    mean_non_empty_dice = (
        sum(non_empty_dice)
        / len(non_empty_dice)
        if non_empty_dice
        else 0.0
    )

    mean_non_empty_iou = (
        sum(non_empty_iou)
        / len(non_empty_iou)
        if non_empty_iou
        else 0.0
    )

    return {
        "samples": len(dataset),
        "empty": empty_count,
        "non_empty": len(non_empty_dice),
        "dice": mean_dice,
        "iou": mean_iou,
        "non_empty_dice": mean_non_empty_dice,
        "non_empty_iou": mean_non_empty_iou,
    }


def main():

    device = torch.device("cpu")

    test_dataset = CrackDataset(
        root_dir=DATASET_ROOT,
        split="test",
        image_size=IMAGE_SIZE,
        augment=False,
    )

    model = UNet().to(device)

    checkpoint_path = os.path.join(
        CHECKPOINT_DIR,
        "best_model.pth",
    )

    model.load_state_dict(
        torch.load(
            checkpoint_path,
            map_location=device,
        )
    )

    model.eval()

    # Group test samples by dataset source.
    source_indices = {
        "S2DS": [],
        "RUB": [],
        "Crack-Seg": [],
    }

    for index, (image_path, _) in enumerate(
        test_dataset.samples
    ):

        source = get_source(
            image_path.name
        )

        if source in source_indices:
            source_indices[source].append(index)

    print()
    print("Test samples by source")
    print("----------------------")

    for source, indices in source_indices.items():

        print(
            f"{source}: {len(indices)}"
        )

    print()

    print("Source-wise evaluation")
    print("----------------------")

    for source, indices in source_indices.items():

        subset = Subset(
            test_dataset,
            indices,
        )

        results = evaluate_source(
            model,
            subset,
            device,
        )

        print()
        print(source)
        print("-" * len(source))

        print(
            f"Samples:          {results['samples']}"
        )

        print(
            f"Empty masks:      {results['empty']}"
        )

        print(
            f"Non-empty masks:  {results['non_empty']}"
        )

        print(
            f"Dice (all):       {results['dice']:.4f}"
        )

        print(
            f"IoU (all):        {results['iou']:.4f}"
        )

        print(
            f"Dice (non-empty): {results['non_empty_dice']:.4f}"
        )

        print(
            f"IoU (non-empty):  {results['non_empty_iou']:.4f}"
        )


if __name__ == "__main__":
    main()