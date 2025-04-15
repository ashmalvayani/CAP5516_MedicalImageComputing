#!/bin/bash -l

#SBATCH --gres=gpu:ampere:1
#SBATCH --ntasks-per-node=1
#SBATCH --constraint=[gmem80]
#SBATCH -c 8
#SBATCH --mem-per-cpu=10G
#SBATCH -p gpu
#SBATCH --error=outs/%J.err
#SBATCH --output=outs/%J.out
#SBATCH --job-name=Assi3_fold1

conda activate medical3

echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICE
nvidia-smi

cd ..
bash train_singlegpu_demo_1.sh