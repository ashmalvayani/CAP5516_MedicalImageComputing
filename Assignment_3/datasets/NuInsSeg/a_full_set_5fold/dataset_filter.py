import os
import pandas as pd

# Set the root directory containing fold_0 to fold_4
root_dir = "/home/ashmal/Courses/MedImgComputing/Assignment_3/finetune-SAM/datasets/NuInsSeg/a_full_set_5fold"
folds = [f"fold_{i}" for i in range(5)]

# Step 1: Load all image-mask pairs for each fold
fold_to_pairs = {}

for fold in folds:
    tissue_path = os.path.join(root_dir, fold, "tissue_images")
    mask_path = os.path.join(root_dir, fold, "binary_masks")

    tissue_files = sorted(os.listdir(tissue_path))
    mask_files = sorted(os.listdir(mask_path))

    # Match files by filename (excluding path)
    matched_pairs = []
    for fname in tissue_files:
        if fname in mask_files:
            tissue_file = os.path.join(f'NuInsSeg/a_full_set_5fold/{fold}/tissue_images', fname)
            mask_file = os.path.join(f'NuInsSeg/a_full_set_5fold/{fold}/binary_masks', fname)
            matched_pairs.append((tissue_file, mask_file))
    fold_to_pairs[fold] = matched_pairs

# Step 2: Create 5 CSV pairs (train + val for each fold)
for i in range(5):
    val_fold = f"fold_{i}"
    train_folds = [f for f in folds if f != val_fold]

    train_data = []
    for tf in train_folds:
        train_data.extend(fold_to_pairs[tf])
    val_data = fold_to_pairs[val_fold]

    # Convert to DataFrame
    train_df = pd.DataFrame(train_data, columns=["image", "mask"])
    val_df = pd.DataFrame(val_data, columns=["image", "mask"])

    # Save to CSV
    train_df.to_csv(f"final_files/train_fold_{i}.csv", index=False, header=False)
    val_df.to_csv(f"final_files/val_fold_{i}.csv", index=False, header=False)

    print(f"Saved train_fold_{i}.csv and val_fold_{i}.csv")
