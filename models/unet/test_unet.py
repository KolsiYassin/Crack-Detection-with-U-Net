import torch

from models.unet.unet import UNet


def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    model = UNet().to(device)

    x = torch.randn(
        2,
        3,
        512,
        512,
    ).to(device)

    y = model(x)

    print("Input shape :", x.shape)
    print("Output shape:", y.shape)

    print(
        "Parameters:",
        sum(p.numel() for p in model.parameters())
    )


if __name__ == "__main__":
    main()