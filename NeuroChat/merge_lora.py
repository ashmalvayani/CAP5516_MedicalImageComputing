from transformers import AutoTokenizer
from peft import PeftModel
from transformers import Qwen2_5_VLForConditionalGeneration  # assuming this is how you're importing the model
import torch

# Step 1: Load base Qwen2.5-VL model
base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct", 
    torch_dtype=torch.float16,  # or "auto"
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

# Step 2: Load LoRA weights
lora_model = PeftModel.from_pretrained(base_model, "/home/ashmal/Courses/MedImgComputing/NeuroChat/Captions/Qwen2-VL-Finetune/output/lora_vision_test_2")

# Step 3: Merge LoRA weights into the base model (optional if you want to save a standalone model)
merged_model = lora_model.merge_and_unload()

# Optional: save the merged model
merged_model.save_pretrained("/home/ashmal/Courses/MedImgComputing/NeuroChat/Captions/Qwen2-VL-Finetune/output/merged_models/qwen2_sft_full")
tokenizer.save_pretrained("/home/ashmal/Courses/MedImgComputing/NeuroChat/Captions/Qwen2-VL-Finetune/output/merged_models/qwen2_sft_full")