import torch

from models.unet.unet import UNet
from losses.losses import BCEDiceLoss


def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    model = UNet().to(device)

    criterion = BCEDiceLoss().to(device)

    images = torch.randn(
        2,
        3,
        512,
        512,
    ).to(device)

    masks = torch.randint(
        0,
        2,
        (
            2,
            1,
            512,
            512,
        ),
    ).float().to(device)

    logits = model(images)

    loss = criterion(
        logits,
        masks,
    )

    print("Logits shape:", logits.shape)
    print("Masks shape :", masks.shape)
    print("Loss        :", loss.item())


if __name__ == "__main__":
    main()