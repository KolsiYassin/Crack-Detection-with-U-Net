from pathlib import Path
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode


class CrackDataset(Dataset):

    def __init__(
        self,
        root_dir,
        split="train",
        image_size=512,
        augment=False, # Data augmentation
        extra_samples=None, # OmniCrack
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.image_size = image_size
        self.augment = augment

        self.image_dir = self.root_dir / split / "images"
        self.mask_dir = self.root_dir / split / "masks"

        self.samples = self._create_samples()
        if extra_samples is not None:
            self.samples.extend(extra_samples)

    def _create_samples(self):

        image_extensions = {
            ".png",
            ".jpg",
            ".jpeg",
            ".bmp",
            ".tif",
            ".tiff",
        }

        images = {
            path.stem: path
            for path in self.image_dir.iterdir()
            if path.suffix.lower() in image_extensions
        }

        masks = {
            path.stem: path
            for path in self.mask_dir.iterdir()
            if path.suffix.lower() in image_extensions
        }

        common_names = sorted(set(images) & set(masks))

        samples = [
            (images[name], masks[name])
            for name in common_names
        ]

        if not samples:
            raise RuntimeError(
                f"No image/mask pairs found in {self.split}"
            )

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        image_path, mask_path = self.samples[index]

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # Resize
        image = TF.resize(
            image,
            [self.image_size, self.image_size],
            interpolation=InterpolationMode.BILINEAR,
        )

        mask = TF.resize(
            mask,
            [self.image_size, self.image_size],
            interpolation=InterpolationMode.NEAREST,
        )

        # Data augmentation
        if self.augment:
            # Horizontal flip
            if random.random() < 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

            # Vertical flip
            if random.random() < 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask)

            # Rotation
            if random.random() < 0.5:
                angle = random.uniform(-15, 15)

                image = TF.rotate(
                    image,
                    angle,
                    interpolation=InterpolationMode.BILINEAR,
                )

                mask = TF.rotate(
                    mask,
                    angle,
                    interpolation=InterpolationMode.NEAREST,
                )

        # Convert image to tensor
        image = TF.to_tensor(image)

        # Convert mask to tensor
        mask = torch.from_numpy(
            np.array(mask)
        ).float()

        # Convert:
        #
        # 0   -> 0
        # 255 -> 1
        #
        mask = mask / 255.0

        # H,W -> 1,H,W
        mask = mask.unsqueeze(0)

        return image, mask