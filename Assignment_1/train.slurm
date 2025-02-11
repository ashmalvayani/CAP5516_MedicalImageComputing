#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=12G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --ntasks-per-node=1
#SBATCH --error=outs/%J.err
#SBATCH --output=outs/%J.out
#SBATCH --job-name=PA1_Scratch_cw

echo "Slurm nodes assigned :$SLURM_JOB_NODELIST"

module purge
module load cuda/12.1 
conda activate llava2

cd /home/ashmal/Courses/MedImgComputing/Assignment_1/A1
python train_individual.py --mode scratch --learning_rate 0.001 --batch_size 32 --num_epochs 40 --model_save_path model_Resnet18_Scratch_cw.pth --log_dir logs/output_ResNet18_Scratch_cw.txt