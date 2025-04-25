#!/bin/bash
#SBATCH --time=7-00:00:00
#SBATCH --cpus-per-gpu=8
#SBATCH --gres=gpu:1
#SBATCH --constraint="gmem48"
#SBATCH --error=outs/Gemma3_btmri-%J.err
#SBATCH --output=outs/Gemma3_btmri-%J.out
#SBATCH --job-name=Gemma3_btmri

## module load cuda/12.4 anaconda3/2022.05
## source activate /home/ashmal/anaconda3/envs/multi_bias

echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICE
nvidia-smi

cd ..

## InternVL, Qwen, LlavaOneVision, MLlama
MODEL_NAME="Gemma3"
PARTITION="btmri"
BATCH_SIZE=256
INPUT_FILE="/home/ashmal/Courses/MedImgComputing/NeuroChat/Captions/Qwen2-VL-Finetune/data/1_brain_tumor/qs_test_${PARTITION}.csv"
OUTPUT_PATH="/home/ashmal/Courses/MedImgComputing/NeuroChat/Captions/Qwen2-VL-Finetune/Evaluation/Outputs/${MODEL_NAME}/${PARTITION}"

python run_inference_batch.py --model_name ${MODEL_NAME} --input_test_file "${INPUT_FILE}" --output_path "${OUTPUT_PATH}" --batch_size ${BATCH_SIZE}