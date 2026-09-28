import torch
import torch.nn as nn


class DiceLoss(nn.Module):

    def __init__(self, smooth=1.0):
        super().__init__()

        self.smooth = smooth

    def forward(self, logits, targets):

        # Convert logits to probabilities
        probabilities = torch.sigmoid(logits)

        # Flatten everything except batch dimension
        probabilities = probabilities.view(
            probabilities.size(0),
            -1,
        )

        targets = targets.view(
            targets.size(0),
            -1,
        )

        # Intersection between prediction and target
        intersection = (
            probabilities * targets
        ).sum(dim=1)

        # Dice coefficient
        dice = (
            2.0 * intersection + self.smooth
        ) / (
            probabilities.sum(dim=1)
            + targets.sum(dim=1)
            + self.smooth
        )

        # Convert Dice score to Dice loss
        dice_loss = 1.0 - dice

        return dice_loss.mean()


class BCEDiceLoss(nn.Module):

    def __init__(
        self,
        bce_weight=1.0,
        dice_weight=1.0,
    ):
        super().__init__()

        self.bce = nn.BCEWithLogitsLoss()

        self.dice = DiceLoss()

        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits, targets):

        bce_loss = self.bce(
            logits,
            targets,
        )

        dice_loss = self.dice(
            logits,
            targets,
        )

        total_loss = (
            self.bce_weight * bce_loss
            + self.dice_weight * dice_loss
        )

        return total_loss