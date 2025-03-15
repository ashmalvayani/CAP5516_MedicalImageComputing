#!/bin/bash

#SBATCH --gres=gpu:ampere:1
#SBATCH -C gmem48
#SBATCH -c 8
#SBATCH --mem=88G
#SBATCH -p gpu
#SBATCH --error=pred_outs/fold2_%J.err
#SBATCH --output=pred_outs/fold2_%J.out
#SBATCH --job-name=fold2

module purge
conda activate llassignmentva2

cd ..
python test2.py --k_fold 2 --test_file /home/ashmal/Courses/MedImgComputing/Assignment_2/TransBTS/data_splits/fold_2/test.txt