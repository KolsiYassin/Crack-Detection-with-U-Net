import torch

from pretrained_unet import PretrainedUNet

model = PretrainedUNet()

x = torch.randn(1, 3, 256, 256)

y = model(x)

print("Input shape:", x.shape)
print("Output shape:", y.shape)