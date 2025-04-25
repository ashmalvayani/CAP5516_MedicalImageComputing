# Installation:

1. Run the following commands to install the environment for installing the code. I am using the [Qwen2-VL-Finetune](https://github.com/2U1/Qwen2-VL-Finetune) repository.

```shell
# Create the environment named 'assignment'
conda create -n neurochat python=3.10 -y

# Activate the environment
conda activate neurochat

# Install the packages from the requirements.txt file in the repository
pip install -r requirements.txt
```

# Brain Tumor Dataset Sources

The dataset for brain tumor classification can be downloaded from any of the following sources:

1. **Brain Tumor Dataset (Figshare)**  
   [https://figshare.com/articles/dataset/brain_tumor_dataset/1512427](https://figshare.com/articles/dataset/brain_tumor_dataset/1512427)

2. **Brain Tumor Classification (MRI) – Kaggle**  
   [https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri)

3. **Brain Tumor MRI Dataset – Kaggle**  
   [https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

---

### Instructions

After downloading the dataset from one of the sources above, transfer it to your cluster and place the files under the `datasets/` directory.

# Generating QA Pairs

Under the following [directory](https://github.com/ashmalvayani/CAP5516_MedicalImageComputing/tree/main/NeuroChat/data/Prepare_Dataset), we have written separate codes for generating QA pairs for each of the three above datasets.

```shell
cd data/Prepare_Dataset

cd 1_brain_tumor
# To prepare batch files
python generate_captions.py
# To upload the files to submit to AzureAI
python submit_file.py
# To submit the batch jobs
python submit_job.py
# To retrieve the results
python retrieve.py

# run the same above files in each folder
cd 2_brain_tumor_classification
cd 3_brain_tunor_classification_MRI
```

For generating tripet questions for each image, run the following file for generating questions.

```shell
cd data/Prepare_Dataset/total_captions
python generate_qas.py

# We have the outputs in Prepare_dataset/total_captions/VQAs/Melted
# and normal version inside Prepare_dataset/total_captions/VQAs/Normal
```

## Dataset Preparation

The script requires a dataset formatted according to the LLaVA specification. The dataset should be a JSON file where each entry contains information about conversations and images. Ensure that the image paths in the dataset match the provided `--image_folder`.<br>

**When using a multi-image dataset, the image tokens should all be `<image>`, and the image file names should have been in a list.**<br><br>
**Please see the example below and follow format your data.**

<summary>Example for single image dataset</summary>

```json
[
  {
    "id": "000000033471",
    "image": "000000033471.jpg",
    "conversations": [
      {
        "from": "human",
        "value": "<image>\nWhat are the colors of the bus in the image?"
      },
      {
        "from": "gpt",
        "value": "The bus in the image is white and red."
      },
      {
        "from": "human",
        "value": "What feature can be seen on the back of the bus?"
      }
    ]
  }
  ...
]
```

# Model Training

**Note:** Deepspeed zero2 is faster than zero3, however it consumes more memory. Also, most of the time zero2 is more stable than zero3.<br><br>
**Tip:** You could use `adamw_bnb_8bit` for optimizer to save memory.

To run the training script, use the following command:

```shell
sbatch job.slurm
```

**Note:** The above code runs on 4 GPUs of 48GB. You can modify as per your GPU resources or you can also run the following command below:

<summary>Click to view full script</summary>

```bash
#!/bin/bash

# Choose your model variant (2B, 3B, or 7B)
# MODEL_NAME="Qwen/Qwen2-VL-7B-Instruct"
# MODEL_NAME="Qwen/Qwen2-VL-2B-Instruct"
# MODEL_NAME="Qwen/Qwen2.5-VL-3B-Instruct"
MODEL_NAME="Qwen/Qwen2.5-VL-7B-Instruct"

# Set Python path
export PYTHONPATH=src:$PYTHONPATH

# Training batch settings
GLOBAL_BATCH_SIZE=4
BATCH_PER_DEVICE=1
NUM_DEVICES=4
GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / (BATCH_PER_DEVICE * NUM_DEVICES)))

# Launch training with DeepSpeed
deepspeed src/training/train.py \
    --use_liger True \
    --lora_enable True \
    --vision_lora True \
    --use_dora False \
    --lora_namespan_exclude "['lm_head', 'embed_tokens']" \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --num_lora_modules -1 \
    --deepspeed scripts/zero3.json \
    --model_id $MODEL_NAME \
    --data_path /home/ashmal/Courses/MedImgComputing/NeuroChat/Captions/Qwen2-VL-Finetune/data/1_brain_tumor/NeuroChat_train.json \
    --image_folder '/home/parthpk/NeuroChat' \
    --remove_unused_columns False \
    --freeze_vision_tower True \
    --freeze_llm True \
    --tune_merger False \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 False \
    --output_dir output/lora_vision_test_2 \
    --num_train_epochs 1 \
    --per_device_train_batch_size $BATCH_PER_DEVICE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --image_min_pixels $((256 * 28 * 28)) \
    --image_max_pixels $((1280 * 28 * 28)) \
    --learning_rate 2e-4 \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --gradient_checkpointing False \
    --report_to tensorboard \
    --lazy_preprocess True \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 10 \
    --dataloader_num_workers 4
```

# For evaluation, we have created a separate test set in
```shell
cd data/1_brain_tumor
# qs_test_bbtest.csv
# qs_test_btcmri.csv
# qs_test_btmri.csv
```

All the job files are placed under jobs folder:
```shell
cd Evaluation/jobs

# For each model, run the following files:
## For NeuroChat
sbatch NeuroChat_bb.sh
sbatch NeuroChat_btmri.sh
sbatch NeuroChat_btcmri.sh

## For Qwen-2.5-VL
sbatch NeuroChat_bb.sh
sbatch NeuroChat_btmri.sh
sbatch NeuroChat_btcmri.sh
..
```
