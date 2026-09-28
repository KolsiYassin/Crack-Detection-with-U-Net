import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
    ):
        super().__init__()

        self.up = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2,
        )

        self.conv = DoubleConv(
            out_channels + skip_channels,
            out_channels,
        )

    def forward(self, x, skip):

        x = self.up(x)

        x = torch.cat(
            [x, skip],
            dim=1,
        )

        x = self.conv(x)

        return x


class PretrainedUNet(nn.Module):

    def __init__(self):
        super().__init__()

        backbone = resnet18(
            weights=ResNet18_Weights.DEFAULT
        )

        # Encoder

        self.encoder1 = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
        )

        self.pool1 = backbone.maxpool

        self.encoder2 = backbone.layer1
        self.encoder3 = backbone.layer2
        self.encoder4 = backbone.layer3
        self.encoder5 = backbone.layer4

        # Decoder

        self.up4 = UpBlock(
            in_channels=512,
            skip_channels=256,
            out_channels=256,
        )

        self.up3 = UpBlock(
            in_channels=256,
            skip_channels=128,
            out_channels=128,
        )

        self.up2 = UpBlock(
            in_channels=128,
            skip_channels=64,
            out_channels=64,
        )

        self.up1 = UpBlock(
            in_channels=64,
            skip_channels=64,
            out_channels=32,
        )

        self.final_up = nn.ConvTranspose2d(
            32,
            16,
            kernel_size=2,
            stride=2,
        )

        self.final_conv = nn.Conv2d(
            16,
            1,
            kernel_size=1,
        )
        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
        )

        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
        )

    def forward(self, x):

        x = (x - self.mean) / self.std
        # Encoder

        x1 = self.encoder1(x)
        # 256 -> 128

        x2 = self.pool1(x1)
        x2 = self.encoder2(x2)
        # 128 -> 64

        x3 = self.encoder3(x2)
        # 64 -> 32

        x4 = self.encoder4(x3)
        # 32 -> 16

        x5 = self.encoder5(x4)
        # 16 -> 8

        # Decoder

        x = self.up4(x5, x4)
        # 8 -> 16

        x = self.up3(x, x3)
        # 16 -> 32

        x = self.up2(x, x2)
        # 32 -> 64

        x = self.up1(x, x1)
        # 64 -> 128

        x = self.final_up(x)
        # 128 -> 256

        x = self.final_conv(x)

        return x