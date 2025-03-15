#!/bin/bash

#SBATCH --gres=gpu:ampere:1
#SBATCH -C gmem80
#SBATCH -c 8
#SBATCH --mem=88G
#SBATCH -p gpu
#SBATCH --error=outs/fold5_%J.err
#SBATCH --output=outs/fold5_%J.out
#SBATCH --job-name=fold5

module purge
conda activate assignment

cd ..
python train.py --k_fold 5 --train_file /home/ashmal/Courses/MedImgComputing/Assignment_2/TransBTS/data_splits/fold_5/train.txt