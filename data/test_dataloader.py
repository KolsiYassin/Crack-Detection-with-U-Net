import torch
from torch.utils.data import DataLoader

from data.dataset import CrackDataset


DATASET_ROOT = "F:/Downloads/Damage_detection/Damage_detection/Brix_dataset"


def main():

    train_dataset = CrackDataset(
        root_dir=DATASET_ROOT,
        split="train",
        image_size=512,
        augment=True
    )

    val_dataset = CrackDataset(
        root_dir=DATASET_ROOT,
        split="val",
        image_size=512,
        augment=False
    )

    test_dataset = CrackDataset(
        root_dir=DATASET_ROOT,
        split="test",
        image_size=512,
        augment=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0,
    )

    print("Dataset sizes:")
    print(f"Train: {len(train_dataset)}")
    print(f"Val:   {len(val_dataset)}")
    print(f"Test:  {len(test_dataset)}")

    images, masks = next(iter(train_loader))

    print("\nBatch:")
    print(f"Images shape: {images.shape}")
    print(f"Masks shape:  {masks.shape}")

    print("\nImage:")
    print(f"Min: {images.min()}")
    print(f"Max: {images.max()}")

    print("\nMask:")
    print(f"Min: {masks.min()}")
    print(f"Max: {masks.max()}")

    print("\nMask unique values:")
    print(torch.unique(masks))


if __name__ == "__main__":
    main()