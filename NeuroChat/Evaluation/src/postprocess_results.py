import os
import pandas as pd

# Define base directories
outputs_base = "/home/ashmal/Courses/MedImgComputing/NeuroChat/Captions/Qwen2-VL-Finetune/Evaluation/Outputs"
ground_truth_files = {
    "btcmri": "/home/ashmal/Courses/MedImgComputing/NeuroChat/Captions/Qwen2-VL-Finetune/data/1_brain_tumor/qs_test_btcmri.csv",
    "btmri": "/home/ashmal/Courses/MedImgComputing/NeuroChat/Captions/Qwen2-VL-Finetune/data/1_brain_tumor/qs_test_btmri.csv"
}
output_gt_dir = "/home/ashmal/Courses/MedImgComputing/NeuroChat/Captions/Qwen2-VL-Finetune/Evaluation/Outputs_GT"
os.makedirs(output_gt_dir, exist_ok=True)

# Loop through each model folder
model_folders = [f for f in os.listdir(outputs_base) if os.path.isdir(os.path.join(outputs_base, f)) and not f.startswith('.')]

print(model_folders)

log = []
for model in model_folders:
    for subset in ["btcmri", "btmri"]:
        try:
            pred_path = os.path.join(outputs_base, model, subset, "results.csv")
            gt_path = ground_truth_files[subset]

            if os.path.exists(pred_path) and os.path.exists(gt_path):
                pred_df = pd.read_csv(pred_path)
                gt_df = pd.read_csv(gt_path)

                # Rename row_idx to id for merging
                if 'row_idx' in pred_df.columns:
                    pred_df = pred_df.rename(columns={'row_idx': 'id'})

                # Merge with suffixes to preserve both 'question' columns
                merged_df = pd.merge(pred_df, gt_df[['id', 'answer', 'question']], on='id', how='left', suffixes=('_pred', '_gt'))

                # Rename the two question columns
                merged_df = merged_df.rename(columns={
                    'question_pred': 'question_prompt',
                    'question_gt': 'gt_question',
                    'answer': 'gt_answer'
                })

                # Reorder and keep relevant columns
                merged_df = merged_df[["id", "image_path", "question_type", "question_prompt", "gt_question", "gt_answer", "model_output"]]

                # Save to Outputs_GT directory
                save_path = os.path.join(output_gt_dir, f"{model}_{subset}_with_gt.csv")
                merged_df.to_csv(save_path, index=False)

                log.append(f"Processed: {model}/{subset}")
            else:
                log.append(f"Missing file for: {model}/{subset}")
        except Exception as e:
            log.append(f"Error processing {model}/{subset}: {e}")

print(log[:10])  # Show a sample of the log
