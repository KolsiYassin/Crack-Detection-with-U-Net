import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import CrackDataset
from models.unet.unet import UNet
from models.ResNet18.pretrained_unet import PretrainedUNet
from config import (
    DATASET_ROOT,
    IMAGE_SIZE,
    BATCH_SIZE,
    NUM_WORKERS,
    CHECKPOINT_DIR,
)


def main():

    device = torch.device("cpu")
    print("Device:", device)

    test_dataset = CrackDataset(
        root_dir=DATASET_ROOT,
        split="test",
        image_size=IMAGE_SIZE,
        augment=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    print("Test samples:", len(test_dataset))

    model = UNet().to(device)  #baseline
    #model = PretrainedUNet().to(device)

    checkpoint_path = CHECKPOINT_DIR / "best_model.pth"

    model.load_state_dict(
        torch.load(
            checkpoint_path,
            map_location=device,
        )
    )

    model.eval()

    # Evaluation
    total_intersection = 0.0
    total_prediction = 0.0
    total_target = 0.0
    total_union = 0.0

    with torch.no_grad():

        for images, masks in tqdm(
            test_loader,
            desc="Testing",
        ):

            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)

            probabilities = torch.sigmoid(logits)

            predictions = (
                probabilities > 0.5
            ).float()

            predictions = predictions.view(
                predictions.size(0), -1
            )

            masks = masks.view(
                masks.size(0), -1
            )

            intersection = (
                predictions * masks
            ).sum()

            prediction_area = predictions.sum()
            target_area = masks.sum()

            union = (
                predictions
                + masks
                - predictions * masks
            ).sum()

            total_intersection += intersection.item()
            total_prediction += prediction_area.item()
            total_target += target_area.item()
            total_union += union.item()

    # Metrics
    dice = (
        2 * total_intersection
        / (
            total_prediction
            + total_target
        )
    )

    iou = (
        total_intersection
        / total_union
    )

    print()
    print("Test Results")
    print("--------------------")
    print(f"Dice: {dice:.4f}")
    print(f"IoU:  {iou:.4f}")


if __name__ == "__main__":
    main()