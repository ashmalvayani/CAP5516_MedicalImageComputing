#!/bin/bash

# Set CUDA device
# export CUDA_VISIBLE_DEVICES="5"

# Define variables
arch="vit_b"  # Change this value as needed
finetune_type="lora"
dataset_name="NuInsSeg"  # Assuming you set this if it's dynamic
targets='combine_all' # make it as binary segmentation 'multi_all' for multi cls segmentation
# Construct train and validation image list paths
img_folder="/home/ashmal/Courses/MedImgComputing/Assignment_3/finetune-SAM/datasets"  # Assuming this is the folder where images are stored
train_img_list="/home/ashmal/Courses/MedImgComputing/Assignment_3/finetune-SAM/datasets/NuInsSeg/a_full_set_5fold/final_files/train_fold_2.csv"
val_img_list="/home/ashmal/Courses/MedImgComputing/Assignment_3/finetune-SAM/datasets/NuInsSeg/a_full_set_5fold/final_files/val_fold_2.csv"


# Construct the checkpoint directory argument
# dir_checkpoint="${finetune_type}_${dataset_name}"

dir_checkpoint="NuInsSeg_Model_Outputs/fold_2"

# Run the Python script
python SingleGPU_train_finetune_noprompt.py \
    -if_warmup True \
    -finetune_type "$finetune_type" \
    -arch "$arch" \
    -if_mask_decoder_adapter True \
    -img_folder "$img_folder" \
    -mask_folder "$img_folder" \
    -sam_ckpt "sam_vit_b_01ec64.pth" \
    -dataset_name "$dataset_name" \
    -dir_checkpoint "$dir_checkpoint" \
    -train_img_list "$train_img_list" \
    -val_img_list "$val_img_list" \
    -b 10 \
    -epochs 500 \
    -if_update_encoder True \
    -if_encoder_lora_layer True \
    -if_decoder_lora_layer True 