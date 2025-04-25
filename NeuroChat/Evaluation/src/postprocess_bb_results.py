import os
import pandas as pd
import re

def extract_bb(text):
    if isinstance(text, str):
        match = re.search(r"\[([^\[\]]+)\]", text)
        if match:
            return f"[{match.group(1)}]"
    return "The bounding box does not exist."


# Define base directories
outputs_base = "/home/ashmal/Courses/MedImgComputing/NeuroChat/Captions/Qwen2-VL-Finetune/Evaluation/Outputs"
ground_truth_files = {
    "bbtest": "/home/ashmal/Courses/MedImgComputing/NeuroChat/Captions/Qwen2-VL-Finetune/data/1_brain_tumor/qs_test_bbtest.csv"
}
output_gt_dir = "/home/ashmal/Courses/MedImgComputing/NeuroChat/Captions/Qwen2-VL-Finetune/Evaluation/Outputs_BB"
os.makedirs(output_gt_dir, exist_ok=True)

# Loop through each model folder
model_folders = [f for f in os.listdir(outputs_base) if os.path.isdir(os.path.join(outputs_base, f)) and not f.startswith('.')]

print(model_folders)

log = []
for model in model_folders:
    for subset in ["bbtest"]:
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
                merged_df = pd.merge(
                    pred_df, 
                    gt_df[['id', 'answer', 'question']], 
                    on='id', 
                    how='left', 
                    suffixes=('_pred', '_gt')
                )

                # Rename columns
                merged_df = merged_df.rename(columns={
                    'question_pred': 'question_prompt',
                    'question_gt': 'gt_question',
                    'answer': 'gt_answer'
                })

                # Add flags for list-format detection
                merged_df["bb_gt_present"] = merged_df["gt_answer"].apply(
                    lambda x: isinstance(x, str) and x.strip().startswith("[") and x.strip().endswith("]")
                )
                # merged_df["bb_pred_present"] = merged_df["model_output"].apply(
                #     lambda x: isinstance(x, str) and x.strip().startswith("[") and x.strip().endswith("]")
                # )


                # merged_df["bb_pred_present"] = merged_df["model_output"].apply(is_list_like_json_answer)

                merged_df["bb_pred_present"] = merged_df["model_output"].apply(
                    lambda x: isinstance(x, str) and "[" in x and "]" in x
                )

                merged_df["filtered_bb_pred"] = merged_df["model_output"].apply(extract_bb)

                # Reorder and keep relevant columns
                merged_df = merged_df[[
                    "id", "image_path", "question_type", "question_prompt",
                    "gt_question", "gt_answer", "model_output",
                    "bb_gt_present", "bb_pred_present", "filtered_bb_pred"
                ]]

                # Save to Outputs_BB directory
                save_path = os.path.join(output_gt_dir, f"{model}_{subset}_with_gt.csv")
                merged_df.to_csv(save_path, index=False)

                log.append(f"Processed: {model}/{subset}")
            else:
                log.append(f"Missing file for: {model}/{subset}")
        except Exception as e:
            log.append(f"Error processing {model}/{subset}: {e}")

print(log[:10])  # Show a sample of the log
