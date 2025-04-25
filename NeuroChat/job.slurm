#!/bin/bash -l

#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --constraint=[gmem48]
#SBATCH -c 12
#SBATCH --mem-per-cpu=10G
#SBATCH -p gpu
#SBATCH --error=outs/%J.err
#SBATCH --output=outs/%J.out
#SBATCH --job-name=NeuroChat

conda activate neurochat

echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICE
nvidia-smi

bash scripts/finetune_lora_vision.sh