#!/bin/bash

#SBATCH --gres=gpu:ampere:1
#SBATCH -C gmem80
#SBATCH -c 8
#SBATCH --mem=88G
#SBATCH -p gpu
#SBATCH --error=outs/fold1_%J.err
#SBATCH --output=outs/fold1_%J.out
#SBATCH --job-name=fold1

module purge
conda activate assignment

cd ..
python train.py --k_fold 1 --train_file /home/ashmal/Courses/MedImgComputing/Assignment_2/TransBTS/data_splits/fold_1/train.txt