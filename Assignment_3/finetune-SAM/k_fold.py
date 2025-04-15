import os
import random
from pathlib import Path
from sklearn.model_selection import KFold

# Define paths
data_files = "datasets/NuInsSeg"
output_dir = "datasets/NuInsSeg/KFolds"

# Create output directory if not exists
os.makedirs(output_dir, exist_ok=True)

# List all files
files = os.listdir(data_files)
random.shuffle(files)  # Shuffle for randomness

# Perform 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for fold_idx, (train_idx, test_idx) in enumerate(kf.split(files), start=1):
    fold_dir = Path(output_dir) / f"fold_{fold_idx}"
    os.makedirs(fold_dir, exist_ok=True)

    train_files = [files[i] for i in train_idx]
    test_files = [files[i] for i in test_idx]

    # Save train and test files
    with open(fold_dir / "train.txt", "w") as f:
        f.writelines("\n".join(train_files))

    with open(fold_dir / "test.txt", "w") as f:
        f.writelines("\n".join(test_files))

print("5-fold data splits have been successfully created!")
