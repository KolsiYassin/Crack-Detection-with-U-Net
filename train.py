import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import CrackDataset
from models.unet.unet import UNet
from models.ResNet18.pretrained_unet import PretrainedUNet
from losses.losses import BCEDiceLoss, DiceLoss
from metrics.metrics import dice_score, iou_score
from config import (
    DATASET_ROOT,
    IMAGE_SIZE,
    BATCH_SIZE,
    LEARNING_RATE,
    NUM_EPOCHS,
    NUM_WORKERS,
    CHECKPOINT_DIR,
    OMNICRACK_SIZE
)

from experiments.experiment_2.create_dataset import load_omnicrack_subset


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
):

    model.train()

    running_loss = 0.0
    running_dice = 0.0
    running_iou = 0.0

    progress_bar = tqdm(
        loader,
        desc="Training",
    )

    for images, masks in progress_bar:

        images = images.to(device)
        masks = masks.to(device)

        # Forward pass
        logits = model(images)

        # Calculate loss
        loss = criterion(
            logits,
            masks,
        )

        # Clear old gradients
        optimizer.zero_grad()

        # Backpropagation
        loss.backward()

        # Update model weights
        optimizer.step()

        # Metrics
        dice = dice_score(
            logits.detach(),
            masks,
        )

        iou = iou_score(
            logits.detach(),
            masks,
        )

        running_loss += loss.item()
        running_dice += dice.item()
        running_iou += iou.item()

        progress_bar.set_postfix(
            loss=loss.item(),
            dice=dice.item(),
            iou=iou.item(),
        )

    num_batches = len(loader)

    return (
        running_loss / num_batches,
        running_dice / num_batches,
        running_iou / num_batches,
    )


def validate(
    model,
    loader,
    criterion,
    device,
):

    model.eval()

    running_loss = 0.0
    running_dice = 0.0
    running_iou = 0.0

    with torch.no_grad():

        progress_bar = tqdm(
            loader,
            desc="Validation",
        )

        for images, masks in progress_bar:

            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)

            loss = criterion(
                logits,
                masks,
            )

            dice = dice_score(
                logits,
                masks,
            )

            iou = iou_score(
                logits,
                masks,
            )

            running_loss += loss.item()
            running_dice += dice.item()
            running_iou += iou.item()

    num_batches = len(loader)

    return (
        running_loss / num_batches,
        running_dice / num_batches,
        running_iou / num_batches,
    )


def main():
    device = torch.device("cpu")

    print("Device:", device)

    #extra_samples = load_omnicrack_subset(OMNICRACK_SIZE) # include omni crack

    train_dataset = CrackDataset(
        root_dir=DATASET_ROOT,
        split="train",
        image_size=IMAGE_SIZE,
        augment=True,
        #extra_samples=extra_samples
    )

    val_dataset = CrackDataset(
        root_dir=DATASET_ROOT,
        split="val",
        image_size=IMAGE_SIZE,
        augment=False,
    )

    print(
        "Training samples:",
        len(train_dataset),
    )

    print(
        "Validation samples:",
        len(val_dataset),
    )

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    # Model
    #model = UNet().to(device) # baseline
    model = PretrainedUNet().to(device) # experiment 4 pretrained 

    # Loss
    #criterion = DiceLoss() # experiment 3, DiceLoss instead of BCEDiceLoss
    criterion = BCEDiceLoss()

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    # Checkpoint directory
    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_val_dice = 0.0

    # Training
    for epoch in range(NUM_EPOCHS):

        print()
        print(
            f"Epoch {epoch + 1}/{NUM_EPOCHS}"
        )

        train_loss, train_dice, train_iou = (
            train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
            )
        )

        val_loss, val_dice, val_iou = (
            validate(
                model,
                val_loader,
                criterion,
                device,
            )
        )

        print()
        print("Results:")
        print(
            f"Train Loss: {train_loss:.4f}"
        )
        print(
            f"Train Dice: {train_dice:.4f}"
        )
        print(
            f"Train IoU:  {train_iou:.4f}"
        )
        print(
            f"Val Loss:   {val_loss:.4f}"
        )
        print(
            f"Val Dice:   {val_dice:.4f}"
        )
        print(
            f"Val IoU:    {val_iou:.4f}"
        )

        # Save best model
        if val_dice > best_val_dice:

            best_val_dice = val_dice

            checkpoint_path = (
                CHECKPOINT_DIR / "best_model.pth"
            )

            torch.save(
                model.state_dict(),
                checkpoint_path,
            )

            print(
                f"Saved best model: "
                f"{checkpoint_path}"
            )


if __name__ == "__main__":
    main()