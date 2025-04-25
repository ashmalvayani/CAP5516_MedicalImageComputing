import os
import json
import pandas as pd

# Re-define paths after state reset
gpt_output_dir = "/home/ashmal/Courses/MedImgComputing/NeuroChat/Captions/Qwen2-VL-Finetune/Evaluation/GPT_Eval_Outputs"
ground_truth_files = {
    "btcmri": "/home/ashmal/Courses/MedImgComputing/NeuroChat/Captions/Qwen2-VL-Finetune/data/1_brain_tumor/qs_test_btcmri.csv",
    "btmri": "/home/ashmal/Courses/MedImgComputing/NeuroChat/Captions/Qwen2-VL-Finetune/data/1_brain_tumor/qs_test_btmri.csv"
}

log = []
for filename in os.listdir(gpt_output_dir):
    if filename.endswith(".json"):
        file_path = os.path.join(gpt_output_dir, filename)

        # Determine subset type from filename
        if "_btcmri" in filename:
            subset = "btcmri"
        elif "_btmri" in filename:
            subset = "btmri"
        else:
            log.append(f"Skipped {filename}: unknown subset")
            continue

        try:
            with open(file_path, 'r') as f:
                eval_data = json.load(f)

            # Load corresponding ground truth
            gt_df = pd.read_csv(ground_truth_files[subset])
            id_to_qtype = dict(zip(gt_df['id'], gt_df['question type']))

            # Add question type to each entry
            for entry in eval_data:
                entry["question type"] = id_to_qtype.get(entry["id"], "unknown")

            # Save updated file
            output_folder_dir = "/home/ashmal/Courses/MedImgComputing/NeuroChat/Captions/Qwen2-VL-Finetune/Evaluation/GPT_Eval_Outputs_GT"
            file = file_path.split('/')[-1]
            output_file_path = f"{output_folder_dir}/{file}"
            
            with open(output_file_path, 'w') as f:
                json.dump(eval_data, f, indent=2)

            log.append(f"Updated: {filename}")
        except Exception as e:
            log.append(f"Error processing {filename}: {e}")

print(log[:10])