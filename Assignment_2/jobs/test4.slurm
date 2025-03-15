#!/bin/bash

#SBATCH --gres=gpu:ampere:1
#SBATCH -C gmem48
#SBATCH -c 8
#SBATCH --mem=88G
#SBATCH -p gpu
#SBATCH --error=pred_outs/fold4_%J.err
#SBATCH --output=pred_outs/fold4_%J.out
#SBATCH --job-name=fold4

module purge
conda activate assignment

cd ..
python test2.py --k_fold 4 --test_file /home/ashmal/Courses/MedImgComputing/Assignment_2/TransBTS/data_splits/fold_4/test.txt