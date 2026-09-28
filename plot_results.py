from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# Paths
ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"

RESULTS_DIR.mkdir(exist_ok=True)


# Training histories
baseline = [
    [1, 0.5831, 0.5376, 0.4134, 0.5420, 0.5299, 0.4197],
    [10, 0.4185, 0.6503, 0.5258, 0.4678, 0.6017, 0.4896],
]

dice_only = [
    [1, 0.5111, 0.5040, 0.3836, 0.4969, 0.5035, 0.3902],
    [2, 0.4056, 0.5954, 0.4674, 0.4759, 0.5243, 0.4083],
    [3, 0.4030, 0.5979, 0.4711, 0.5007, 0.4995, 0.3879],
    [4, 0.3852, 0.6152, 0.4875, 0.4359, 0.5621, 0.4498],
    [5, 0.3738, 0.6261, 0.4996, 0.4496, 0.5502, 0.4409],
    [6, 0.3742, 0.6257, 0.4996, 0.4283, 0.5699, 0.4631],
    [7, 0.4104, 0.5896, 0.4684, 0.4421, 0.5575, 0.4472],
    [8, 0.3878, 0.6125, 0.4880, 0.4277, 0.5722, 0.4595],
    [9, 0.3581, 0.6418, 0.5157, 0.4167, 0.5823, 0.4702],
    [10, 0.4238, 0.5763, 0.4576, 0.4564, 0.5434, 0.4297],
]

resnet18 = [
    [1, 0.5670, 0.5305, 0.4188, 0.5054, 0.5659, 0.4545],
    [2, 0.4285, 0.6377, 0.5110, 0.4839, 0.5764, 0.4659],
    [3, 0.4176, 0.6447, 0.5187, 0.4853, 0.5820, 0.4716],
    [4, 0.4139, 0.6489, 0.5234, 0.4909, 0.5719, 0.4619],
    [5, 0.4084, 0.6520, 0.5270, 0.4886, 0.5777, 0.4676],
    [6, 0.4086, 0.6529, 0.5283, 0.4704, 0.5974, 0.4851],
    [7, 0.4134, 0.6501, 0.5252, 0.4667, 0.6039, 0.4927],
    [8, 0.4072, 0.6565, 0.5325, 0.4335, 0.6291, 0.5216],
    [9, 0.4022, 0.6612, 0.5371, 0.4214, 0.6375, 0.5287],
    [10, 0.4115, 0.6519, 0.5273, 0.4629, 0.6129, 0.5036],
]

COLUMNS = [
    "epoch",
    "train_loss",
    "train_dice",
    "train_iou",
    "val_loss",
    "val_dice",
    "val_iou",
]


# Save training histories
pd.DataFrame(
    baseline,
    columns=COLUMNS,
).to_csv(
    RESULTS_DIR / "baseline_available.csv",
    index=False,
)

pd.DataFrame(
    dice_only,
    columns=COLUMNS,
).to_csv(
    RESULTS_DIR / "dice_only.csv",
    index=False,
)

pd.DataFrame(
    resnet18,
    columns=COLUMNS,
).to_csv(
    RESULTS_DIR / "pretrained_resnet18.csv",
    index=False,
)


# Final experiment summary
summary = pd.DataFrame(
    [
        [
            "Baseline U-Net",
            4539,
            "BCE + Dice",
            "U-Net",
            0.6017,
            0.4896,
            0.6525,
            0.4843,
        ],
        [
            "+3000 OmniCrack",
            7539,
            "BCE + Dice",
            "U-Net",
            0.5924,
            0.4828,
            0.6286,
            0.4583,
        ],
        [
            "Dice-only",
            4539,
            "Dice",
            "U-Net",
            0.5823,
            0.4702,
            0.6272,
            0.4569,
        ],
        [
            "Pretrained ResNet-18",
            4539,
            "BCE + Dice",
            "ResNet-18",
            0.6375,
            0.5287,
            0.6598,
            0.4923,
        ],
    ],
    columns=[
        "experiment",
        "train_samples",
        "loss",
        "encoder",
        "best_val_dice",
        "best_val_iou",
        "test_dice",
        "test_iou",
    ],
)

summary.to_csv(
    RESULTS_DIR / "experiment_summary.csv",
    index=False,
)


# Source-wise evaluation
source = pd.DataFrame(
    [
        ["S2DS", 0.1866, 0.1029],
        ["RUB", 0.6696, 0.5033],
        ["Crack-Seg", 0.7449, 0.5935],
    ],
    columns=[
        "source",
        "dice",
        "iou",
    ],
)

source.to_csv(
    RESULTS_DIR / "source_wise.csv",
    index=False,
)


# Plot 1: Final test-set comparison
plt.figure(figsize=(9, 5))

x = range(len(summary))
width = 0.36

plt.bar(
    [i - width / 2 for i in x],
    summary["test_dice"],
    width,
    label="Dice",
)

plt.bar(
    [i + width / 2 for i in x],
    summary["test_iou"],
    width,
    label="IoU",
)

plt.xticks(
    list(x),
    summary["experiment"],
    rotation=20,
    ha="right",
)

plt.ylabel("Score")
plt.title("Final Test-Set Segmentation Performance")
plt.ylim(0, 0.75)
plt.legend()

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "test_comparison.png",
    dpi=180,
)

plt.close()


# Plot 2: Pretrained ResNet-18 training curves
resnet_df = pd.DataFrame(
    resnet18,
    columns=COLUMNS,
)

plt.figure(figsize=(8, 5))

plt.plot(
    resnet_df["epoch"],
    resnet_df["train_dice"],
    marker="o",
    label="Train Dice",
)

plt.plot(
    resnet_df["epoch"],
    resnet_df["val_dice"],
    marker="o",
    label="Validation Dice",
)

plt.xlabel("Epoch")
plt.ylabel("Dice")
plt.title("Pretrained ResNet-18: Dice During Training")
plt.xticks(range(1, 11))
plt.legend()

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "resnet18_dice_curve.png",
    dpi=180,
)

plt.close()


# Plot 3: Source-wise performance
plt.figure(figsize=(8, 5))

x = range(len(source))
width = 0.36

plt.bar(
    [i - width / 2 for i in x],
    source["dice"],
    width,
    label="Dice",
)

plt.bar(
    [i + width / 2 for i in x],
    source["iou"],
    width,
    label="IoU",
)

plt.xticks(
    list(x),
    source["source"],
)

plt.ylabel("Score")
plt.title("Baseline Performance by Dataset Source")
plt.ylim(0, 0.85)
plt.legend()

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "source_wise_comparison.png",
    dpi=180,
)

plt.close()


print("Results created in:")
print(RESULTS_DIR)

print("\nFiles:")

for path in sorted(RESULTS_DIR.iterdir()):
    print(f" - {path.name}")
