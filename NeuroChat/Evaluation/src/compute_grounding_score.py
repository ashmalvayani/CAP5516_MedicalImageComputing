import pandas as pd
import ast

def compute_iou(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    inter_width = max(0, inter_x_max - inter_x_min)
    inter_height = max(0, inter_y_max - inter_y_min)
    inter_area = inter_width * inter_height

    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = area1 + area2 - inter_area

    return inter_area / union_area if union_area > 0 else 0.0

def scale_box(box, ratio):
    return [coord * ratio for coord in box]

def compute_grounding_score_from_csv(filepath, ratio=1.0):
    df = pd.read_csv(filepath)

    iou_scores = []

    # For binary classification metrics
    tp = 0
    fp = 0
    fn = 0
    tn = 0

    for _, row in df.iterrows():
        gt_present = row['bb_gt_present']
        pred_present = row['bb_pred_present']
        gt_box_raw = str(row['gt_answer']).strip()
        pred_box_raw = str(row['filtered_bb_pred']).strip()

        if gt_present:
            if gt_box_raw == "There is no tumor in this image":
                gt_box = []
            else:
                try:
                    gt_box = ast.literal_eval(gt_box_raw)
                except:
                    gt_box = []

            if not pred_present or pred_box_raw == "The bounding box does not exist.":
                iou_scores.append(0.0)
                fn += 1  # False negative: tumor present, prediction missing
            else:
                try:
                    pred_box = scale_box(ast.literal_eval(pred_box_raw), ratio)
                    iou = compute_iou(gt_box, pred_box)
                    iou_scores.append(iou)
                    tp += 1  # True positive: tumor detected
                except:
                    iou_scores.append(0.0)
                    fn += 1  # Consider failure to parse as FN
        else:
            # No GT box
            if not pred_present or pred_box_raw == "The bounding box does not exist.":
                tn += 1  # Correctly predicted nothing
            else:
                fp += 1  # False positive: predicted tumor when none exists

    average_1 = sum(iou_scores) / len(iou_scores) if iou_scores else 0.0

    # Compute full binary F1 score
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    total = len(iou_scores) + tp + fp + fn + tn
    grounding_score = (
        (len(iou_scores) * average_1 + (tp + fp + fn + tn) * f1_score) / total
        if total > 0 else 0.0
    )

    return {
        "average_1 (IoU)": average_1,
        "average_2 (F1 score overall)": f1_score,
        "grounding_score": grounding_score,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn
    }

# Example usage
model_file = "NeuroChat"
src = f"/home/ashmal/Courses/MedImgComputing/NeuroChat/Captions/Qwen2-VL-Finetune/Evaluation/Outputs_BB/{model_file}_bbtest_with_gt.csv"

result = compute_grounding_score_from_csv(src, ratio=2)
print(model_file, result)
