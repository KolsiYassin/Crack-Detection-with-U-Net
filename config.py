from pathlib import Path

DATASET_ROOT = (
    "F:/Downloads/Damage_detection/"
    "Damage_detection/Brix_dataset"
)

IMAGE_SIZE = 256
BATCH_SIZE = 4


LEARNING_RATE = 1e-3
NUM_EPOCHS = 10

NUM_WORKERS = 0

EXPERIMENT_NAME = "DiceLoss"
OMNICRACK_SIZE = 3000

CHECKPOINT_DIR = Path("checkpoints") / EXPERIMENT_NAME