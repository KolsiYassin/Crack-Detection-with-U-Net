import os

import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from data.dataset import CrackDataset
from models.unet.unet import UNet
from config import (
    DATASET_ROOT,
    IMAGE_SIZE,
    BATCH_SIZE,
    NUM_WORKERS,
    CHECKPOINT_DIR,
)


OUTPUT_DIR = "outputs/predictions"


def main():

    device = torch.device("cpu")

    # -------------------------
    # Dataset
    # -------------------------

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

    # -------------------------
    # Model
    # -------------------------

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

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # -------------------------
    # Get predictions
    # -------------------------

    image_count = 0

    with torch.no_grad():

        for images, masks in test_loader:

            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)

            probabilities = torch.sigmoid(logits)

            predictions = (
                probabilities > 0.5
            ).float()

            # -------------------------
            # Visualize each image
            # -------------------------

            for i in range(images.size(0)):

                image = images[i].cpu().permute(
                    1, 2, 0
                ).numpy()

                mask = masks[i, 0].cpu().numpy()

                prediction = (
                    predictions[i, 0]
                    .cpu()
                    .numpy()
                )

                probability = (
                    probabilities[i, 0]
                    .cpu()
                    .numpy()
                )

                fig, axes = plt.subplots(
                    1,
                    4,
                    figsize=(16, 4),
                )

                axes[0].imshow(image)
                axes[0].set_title("Original")

                axes[1].imshow(
                    mask,
                    cmap="gray",
                )
                axes[1].set_title("Ground Truth")

                axes[2].imshow(
                    prediction,
                    cmap="gray",
                )
                axes[2].set_title("Prediction")

                axes[3].imshow(image)
                axes[3].imshow(
                    prediction,
                    cmap="Reds",
                    alpha=0.4,
                )
                axes[3].set_title("Overlay")

                for ax in axes:
                    ax.axis("off")

                plt.tight_layout()

                output_path = os.path.join(
                    OUTPUT_DIR,
                    f"prediction_{image_count:03d}.png",
                )

                plt.savefig(output_path)
                plt.close()

                image_count += 1

                # Only visualize first 20 images
                if image_count >= 20:
                    print(
                        f"Saved {image_count} predictions "
                        f"to {OUTPUT_DIR}"
                    )
                    return


if __name__ == "__main__":
    main()