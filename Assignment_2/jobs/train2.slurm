#!/bin/bash

#SBATCH --gres=gpu:ampere:1
#SBATCH -C gmem80
#SBATCH -c 8
#SBATCH --mem=88G
#SBATCH -p gpu
#SBATCH --error=outs/fold2_%J.err
#SBATCH --output=outs/fold2_%J.out
#SBATCH --job-name=fold2

module purge
conda activate assignment

cd ..
python train.py --k_fold 2 --train_file /home/ashmal/Courses/MedImgComputing/Assignment_2/TransBTS/data_splits/fold_2/train.txt