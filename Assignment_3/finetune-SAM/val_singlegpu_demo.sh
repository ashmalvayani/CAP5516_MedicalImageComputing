#!/bin/bash

# Set CUDA device
#export CUDA_VISIBLE_DEVICES="5"

# Define variables
arch="vit_b"  # Change this value as needed
finetune_type="lora"
dataset_name="NuInsSeg"  # Assuming you set this if it's dynamic

val_img_list="/home/ashmal/Courses/MedImgComputing/Assignment_3/finetune-SAM/datasets/NuInsSeg/a_full_set_5fold/final_files/val_fold_4.csv"

# Construct the checkpoint directory argument
dir_checkpoint="/home/ashmal/Courses/MedImgComputing/Assignment_3/finetune-SAM/NuInsSeg_Model_Outputs/fold_4"

# Run the Python script
python val_finetune_noprompt.py \
    -if_warmup True \
    -finetune_type "$finetune_type" \
    -arch "$arch" \
    -if_mask_decoder_adapter True \
    -dataset_name "$dataset_name" \
    -dir_checkpoint "$dir_checkpoint" \
    -val_img_list "$val_img_list"
