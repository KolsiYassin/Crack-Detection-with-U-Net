import matplotlib.pyplot as plt

from data.dataset import CrackDataset


DATASET_ROOT = (
    "F:/Downloads/Damage_detection/"
    "Damage_detection/Brix_dataset"
)


def main():

    dataset = CrackDataset(
        root_dir=DATASET_ROOT,
        split="train",
        image_size=512,
        augment=True,
    )

    fig, axes = plt.subplots(
        4,
        4,
        figsize=(12, 12),
    )

    for i in range(4):

        image, mask = dataset[i]

        image = image.permute(1, 2, 0).numpy()
        mask = mask.squeeze(0).numpy()

        axes[i, 0].imshow(image)
        axes[i, 0].set_title("Image")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(mask, cmap="gray")
        axes[i, 1].set_title("Mask")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(image)
        axes[i, 2].imshow(
            mask,
            cmap="Reds",
            alpha=0.4,
        )
        axes[i, 2].set_title("Overlay")
        axes[i, 2].axis("off")

        axes[i, 3].imshow(
            mask,
            cmap="gray",
        )
        axes[i, 3].set_title("Mask")
        axes[i, 3].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()