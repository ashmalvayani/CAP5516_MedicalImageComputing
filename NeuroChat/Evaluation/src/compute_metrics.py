# import os
# import json
# import re
# from collections import defaultdict

# # Define directory containing GPT evaluation files
# gpt_eval_dir = "/home/ashmal/Courses/MedImgComputing/NeuroChat/Captions/Qwen2-VL-Finetune/Evaluation/GPT_Eval_Outputs_GT"

# # Collect scores grouped by model and subset
# model_subset_scores = defaultdict(list)

# # Process each file
# for file_name in os.listdir(gpt_eval_dir):
#     if file_name.endswith(".json"):
#         file_path = os.path.join(gpt_eval_dir, file_name)
#         model_name, subset = file_name.replace(".json", "").split("_", 1)

#         try:
#             with open(file_path, 'r') as f:
#                 data = json.load(f)

#             for item in data:
#                 caption = item.get("caption", "")
#                 # Use regex to extract the score value from string like: {'score': 10, 'reason': '...'}
#                 match = re.search(r"'score':\s*(\d+)", caption)
#                 if match:
#                     score = int(match.group(1))
#                     model_subset_scores[(model_name, subset)].append(score)
#         except Exception as e:
#             model_subset_scores[(model_name, subset)].append(f"Error: {e}")

# # Compute averages
# avg_scores = []
# for (model, subset), scores in model_subset_scores.items():
#     valid_scores = [s for s in scores if isinstance(s, int)]
#     avg = sum(valid_scores) / len(valid_scores) if valid_scores else 0
#     avg_scores.append({
#         "model_name": model,
#         "subset": subset,
#         "average_score": round(avg, 2),
#         "num_samples": len(valid_scores)
#     })

# import pandas as pd
# avg_scores_df = pd.DataFrame(avg_scores)
# print(avg_scores_df)






# import os
# import json
# import re
# from collections import defaultdict
# import pandas as pd

# # Simulate directory (user said files are like in GPT_Eval_Outputs_GT)
# gpt_eval_dir = "/home/ashmal/Courses/MedImgComputing/NeuroChat/Captions/Qwen2-VL-Finetune/Evaluation/GPT_Eval_Outputs_GT"  # Adjusted for uploaded content if needed

# # Store scores by model and question type
# model_qtype_scores = defaultdict(list)

# # Traverse all JSON files
# for filename in os.listdir(gpt_eval_dir):
#     if filename.endswith(".json"):
#         file_path = os.path.join(gpt_eval_dir, filename)
#         try:
#             with open(file_path, 'r') as f:
#                 data = json.load(f)

#             for item in data:
#                 model_name = item.get("model_name", "unknown_model")
#                 question_type = item.get("question type", "unknown_type")
#                 caption = item.get("caption", "")

#                 match = re.search(r"'score':\s*(\d+)", caption)
#                 if match:
#                     score = int(match.group(1))
#                     model_qtype_scores[(model_name, question_type)].append(score)

#         except Exception as e:
#             print(f"Error reading {filename}: {e}")

# # Aggregate and average scores
# rows = []
# for (model, qtype), scores in model_qtype_scores.items():
#     avg_score = round(sum(scores) / len(scores), 2) if scores else 0
#     rows.append({
#         "model_name": model,
#         "question_type": qtype,
#         "average_score": avg_score,
#         "num_samples": len(scores)
#     })

# df_qtype_avg = pd.DataFrame(rows)
# print(df_qtype_avg)

# df_qtype_avg.to_csv("LMM_Scores.csv", index=False)



import os
import json
import re
from collections import defaultdict
import pandas as pd

gpt_eval_dir = "/home/ashmal/Courses/MedImgComputing/NeuroChat/Captions/Qwen2-VL-Finetune/Evaluation/GPT_Eval_Outputs_GT"  # Adjusted for uploaded content if needed

model_qtype_scores = defaultdict(list)

for filename in os.listdir(gpt_eval_dir):
    if filename.endswith(".json"):
        file_path = os.path.join(gpt_eval_dir, filename)
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)

            model_name, subset = filename.replace(".json", "").split("_", 1)

            for item in data:
                question_type = item.get("question type", "unknown_type")
                caption = item.get("caption", "")

                match = re.search(r"'score':\s*(\d+)", caption)
                if match:
                    score = int(match.group(1))
                    model_qtype_scores[(model_name, question_type, subset)].append(score)

        except Exception as e:
            print(f"Error reading {filename}: {e}")

# Aggregate and average scores
rows = []
for (model, qtype, subset), scores in model_qtype_scores.items():
    avg_score = round(sum(scores) / len(scores), 2) if scores else 0
    rows.append({
        "model_name": model,
        "subset": subset,
        "question_type": qtype,
        "average_score": avg_score,
        "num_samples": len(scores)
    })

df_qtype_avg = pd.DataFrame(rows)
print(df_qtype_avg)

# df_qtype_avg.to_csv("LMM_Scores.csv", index=False)


pivot_df = df_qtype_avg.pivot_table(index=["model_name", "subset"], columns="question_type", values="average_score").reset_index()
pivot_df["Avg. Score"] = pivot_df[["tumor_presence", "tumor_type", "image_type"]].mean(axis=1)
# print(pivot_df)

pivot_btcmri = pivot_df[pivot_df['subset'] == 'btcmri'].reset_index(drop=True)
pivot_btmri = pivot_df[pivot_df['subset'] == 'btmri'].reset_index(drop=True)

score_cols = ["caption", "image_type", "tumor_presence", "tumor_type", "Avg. Score"]

pivot_btcmri_scaled = pivot_btcmri.copy()
pivot_btmri_scaled = pivot_btmri.copy()

pivot_btcmri_scaled[score_cols] = pivot_btcmri_scaled[score_cols] * 10
pivot_btmri_scaled[score_cols] = pivot_btmri_scaled[score_cols] * 10

print(pivot_btcmri_scaled)
print(pivot_btmri_scaled)