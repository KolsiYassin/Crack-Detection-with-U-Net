import torch


def dice_score(logits, targets, threshold=0.5):

    probabilities = torch.sigmoid(logits)

    predictions = (
        probabilities > threshold
    ).float()

    predictions = predictions.view(
        predictions.size(0),
        -1,
    )

    targets = targets.view(
        targets.size(0),
        -1,
    )

    intersection = (
        predictions * targets
    ).sum(dim=1)

    dice = (
        2 * intersection + 1e-7
    ) / (
        predictions.sum(dim=1)
        + targets.sum(dim=1)
        + 1e-7
    )

    return dice.mean()


def iou_score(logits, targets, threshold=0.5):

    probabilities = torch.sigmoid(logits)

    predictions = (
        probabilities > threshold
    ).float()

    predictions = predictions.view(
        predictions.size(0),
        -1,
    )

    targets = targets.view(
        targets.size(0),
        -1,
    )

    intersection = (
        predictions * targets
    ).sum(dim=1)

    union = (
        predictions
        + targets
        - predictions * targets
    ).sum(dim=1)

    iou = (
        intersection + 1e-7
    ) / (
        union + 1e-7
    )

    return iou.mean()