#!/usr/bin/env python3
import argparse
import os
import pandas as pd
from io import BytesIO
import torch
import base64
import re
import json
import base64
from PIL import Image
from io import BytesIO
from glob import glob

from vllm import LLM, SamplingParams
from huggingface_hub import login
from transformers import AutoProcessor
from transformers import AutoTokenizer
from qwen_vl_utils import process_vision_info

login(token="")


def get_NeuroChat_vl(parameters='7B', version='2.5'):
    """NeuroChat"""
    # if parameters=='default':
    #     parameters = '7B'
    # if version=='default':
    #     version = '2.5'

    # if version=='2.5':
    #     model_name = f"Qwen/Qwen2.5-VL-{parameters}-Instruct"
    # elif version=='2':
    #     model_name = f"Qwen/Qwen2-VL-{parameters}-Instruct"
    # else:
    #     raise NotImplementedError
    
    model_name = "/home/ashmal/Courses/MedImgComputing/NeuroChat/Captions/Qwen2-VL-Finetune/output/merged_models/qwen2_sft_full"
    
    llm = LLM(
        model=model_name,
        max_model_len=4096,
        max_num_seqs=5,
        disable_mm_preprocessor_cache=True,
        tensor_parallel_size=torch.cuda.device_count(),
        gpu_memory_utilization=0.98,
        trust_remote_code=True,
    )

    processor = AutoProcessor.from_pretrained(model_name)

    return llm, None, processor


def get_qwen_vl(parameters='7B', version='2.5'):
    """Qwen/Qwen2.5-VL-7B-Instruct"""
    if parameters=='default':
        parameters = '7B'
    if version=='default':
        version = '2.5'

    if version=='2.5':
        model_name = f"Qwen/Qwen2.5-VL-{parameters}-Instruct"
    elif version=='2':
        model_name = f"Qwen/Qwen2-VL-{parameters}-Instruct"
    else:
        raise NotImplementedError
    
    llm = LLM(
        model=model_name,
        max_model_len=4096,
        max_num_seqs=5,
        disable_mm_preprocessor_cache=True,
        tensor_parallel_size=torch.cuda.device_count(),
        gpu_memory_utilization=0.98,
        trust_remote_code=True,
    )

    processor = AutoProcessor.from_pretrained(model_name)

    return llm, None, processor

def get_gemma3(parameters='12b', version='3'):
    """Gemma3ForConditionalGeneration"""
    if parameters=='default':
        parameters = '12b'
    if version=='default':
        version = '3'

    parameters = parameters.lower()
    model_name = f"google/gemma-{version}-{parameters}-it"

    llm = LLM(
        model=model_name,
        max_model_len=2048,
        max_num_seqs=2,
        mm_processor_kwargs={"do_pan_and_scan": True},
        tensor_parallel_size=torch.cuda.device_count(),
        gpu_memory_utilization=0.98,
        limit_mm_per_prompt={"image": 1},
    )

    processor = AutoProcessor.from_pretrained(model_name)

    return llm, None, processor

def get_internvl(parameters='8B', version='3'):
    """OpenGVLab/InternVL3-8B"""
    if parameters=='default':
        parameters = '8B'
    if version=='default':
        version = '3'

    model_name = f"OpenGVLab/InternVL{version}-{parameters}"

    llm = LLM(
        model=model_name,
        max_model_len=4096,
        max_num_seqs=5,
        disable_mm_preprocessor_cache=True,
        tensor_parallel_size=torch.cuda.device_count(),
        gpu_memory_utilization=0.98,
        trust_remote_code=True,
    )

    processor = AutoTokenizer.from_pretrained(model_name)

    #stop_tokens = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|end|>"]
    #stop_token_ids = [processor.convert_tokens_to_ids(i) for i in stop_tokens]

    return llm, None, processor

def get_llava_onevision_qwen2(parameters='8B', version='3'):
    """llava-hf/llava-onevision-qwen2-7b-ov-hf"""
    model_name = "llava-hf/llava-onevision-qwen2-7b-ov-hf"
    llm = LLM(
        model=model_name,
        max_model_len=16384,
        tensor_parallel_size=torch.cuda.device_count(),
        gpu_memory_utilization=0.90,
        disable_mm_preprocessor_cache=True,
    )
    # No special stop tokens by default
    stop_token_ids = None

    processor = AutoTokenizer.from_pretrained(model_name)

    return llm, stop_token_ids, processor

def get_mllama(parameters='11B', version='3.2'):
    model_name = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    llm = LLM(model=model_name,
        max_model_len=8192,
        max_num_seqs=2,
        tensor_parallel_size=torch.cuda.device_count(),
        gpu_memory_utilization=0.98,
        limit_mm_per_prompt={"image": 1},
    )

    processor = AutoTokenizer.from_pretrained(model_name)

    return llm, None, processor


SAVE_EVERY = 100

MODEL_MAP = {
    'InternVL': get_internvl,                       # default: 8B, 3
    'Gemma3': get_gemma3,                           # default: 12b, 3
    'Qwen': get_qwen_vl,                            # default: 7B, 2.5
    'LlavaOneVision': get_llava_onevision_qwen2,    # default: 7B, ?
    'MLlama': get_mllama,                           # default: 11B, 3.2
    'NeuroChat': get_NeuroChat_vl,                  # default: 7B
}


def encode_image(image_path):
    try:
        with Image.open(image_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')

            width_percent = 256 / float(img.size[0])
            new_height = int((float(img.size[1]) * width_percent))

            img_resized = img.resize((256, new_height), Image.LANCZOS)

            buffered = BytesIO()
            img_resized.save(buffered, format="PNG")

            return base64.b64encode(buffered.getvalue()).decode('utf-8')
    except Exception as e:
        return None  
    
def build_prompt(model_name, processor, image, question):
    post_prompt = "Give the output in strict JSON format. {\"answer\": \"ANSWER HERE\""
    prompt = question + '\n' + post_prompt

    image_asset = Image.open(image).convert("RGB")
    messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            },
    ]
    

    if model_name.lower() == 'qwen' or model_name.lower() == 'neurochat':
        base64_img = f'data:image;base64,{encode_image(image)}'
        messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": base64_img, "min_pixels": 224 * 224, "max_pixels": 1280 * 28 * 28},
                        {"type": "text", "text": prompt},
                    ],
                },
        ]
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
        mm_data = {"image": image_inputs} if image_inputs is not None else {}

        llm_inputs = {"prompt": prompt, "multi_modal_data": mm_data}

        return llm_inputs
    
    elif model_name.lower() == 'internvl':
        ## test InternVL3
        messages = [{
                'role': 'user',
                'content': f"<image>\n{prompt}"
            }]
        prompts = processor.apply_chat_template(messages,
                                            tokenize=False,
                                            add_generation_prompt=True)
        
        llm_inputs = { "prompt": prompts,
                "multi_modal_data": {"image": image_asset}
        }

        return llm_inputs
    
    elif model_name.lower() == 'llavaonevision':
        prompt_str = f"<|im_start|>user <image>\n{prompt}<|im_end|>" + \
                            "<|im_start|>assistant\n"
        llm_inputs = { "prompt": prompt_str,
                "multi_modal_data": {"image": image_asset}
        }

        return llm_inputs
    
    elif model_name.lower() == 'mllama':
        llm_inputs = processor.apply_chat_template(messages,
                                            add_generation_prompt=True,
                                            tokenize=False)
        return llm_inputs
    
    elif model_name.lower() == 'gemma3':
        prompt_str = ("<bos><start_of_turn>user\n"
                f"<start_of_image>{prompt}<end_of_turn>\n"
                "<start_of_turn>model\n")
        
        llm_inputs = { "prompt": prompt_str,
                "multi_modal_data": {"image": image_asset}
        }

        return llm_inputs
        
    else:
        raise NotImplementedError

def run_inference(model, prompt, sampling_params):
    generated = model.generate([prompt], sampling_params=sampling_params)
    return generated[0].outputs[0].text


def main(args):
    model_name = args.model_name
    version = args.version
    parameters = args.parameters

    build_llm_fn = MODEL_MAP[model_name]
    model, stop_token_ids, processor = build_llm_fn(parameters=parameters, version=version)

    sampling_params = SamplingParams(
        temperature=0.1,        # Deterministic
        max_tokens=256,
        stop_token_ids=stop_token_ids,
    )

    image_folder = '/home/parthpk/NeuroChat/'
    df = pd.read_csv(args.input_test_file)

    batch_inputs = []
    batch_record_map = []  # Keep track of which row & image_col => index in outputs
    results_data = []

    batch_processed = 0
    
    for row_idx, row in df.iterrows():
        img_path = row.get('path')
        image_path = f"{image_folder}/{img_path}"
        
        question_type = row.get('question type')
        question = row.get('question')
        row_idx = row.get('id')
        
        # UID_without_image = row.get('UID_without_image')

        if question_type == 'caption':
            question = (
                f"Provide helpful and detailed answer to the question."
                f"{question}"
            )
        elif question_type == 'ground':
            question = (
                f"If the tumor exists in the image, {question}\nOtherwise, say 'There is no tumor in this image.'"
            )
        else:
            question = (
                f"Provide short and direct answer for the user's question."
                f"{question}"
            )
        # prompt_str = question


        prompt = build_prompt(model_name, processor, image_path, question)

        batch_inputs.append(prompt)
        batch_record_map.append((row_idx, image_path, question_type, question))

        if ((((row_idx+1) % args.batch_size) == 0) or (row_idx==df.shape[0]-1)):
            print(f"Running inference on {len(batch_inputs)} items...")
            outputs = model.generate(batch_inputs, sampling_params=sampling_params)
            batch_processed += 1

            for out_idx, output_obj in enumerate(outputs):
                text_output = output_obj.outputs[0].text

                row_idx, image_path, question_type, question = batch_record_map[out_idx]

                results_data.append({
                    "row_idx": row_idx,
                    "image_path": image_path,
                    "question_type": question_type,
                    "question": question,
                    "model_output": text_output
                })
            
            df_results = pd.DataFrame(results_data)

            # Create output directory.
            # base_output_dir = os.path.join(args.output_path, f"{args.model_name}")
            # os.makedirs(base_output_dir, exist_ok=True)

            # output_path = os.path.join(base_output_dir, f"results.csv")
            
            output_path = args.output_path
            os.makedirs(output_path, exist_ok=True)
            print(f">>> Inference partially done: {len(batch_inputs)}. Saving results to {output_path}")
            # import pdb; pdb.set_trace()
            df_results.to_csv(os.path.join(output_path, 'results.csv'), index=False)

            # Clear for next batch
            batch_inputs = []
            batch_record_map = []

    print(f">> Inference done. Results saved to {output_path}")


def check_pytorch_gpu():
    try:
        if torch.cuda.is_available():
            print(f"PyTorch can access {torch.cuda.device_count()} GPU(s).")
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("PyTorch cannot access any GPUs.")
    except Exception as e:
        print(f"An error occurred: {e}")



if __name__ == "__main__":
    ## Usage: python run_inference_batch.py --model_name InternVL --csv_path "/home/ja339952/CAP6412/temp_context.csv" --batch_size 10
    parser = argparse.ArgumentParser(description="Batch inference for specific V+L models.")
    parser.add_argument("--input_test_file", type=str, default="./",
                        help="File path for type of test set.")
    parser.add_argument("--model_name", type=str, required=False, default = "Qwen2-VL-7B-Instruct",
                        help="Which model to run. Must be one of: " + ", ".join(MODEL_MAP.keys()))
    parser.add_argument("--version", type=str, default="default",
                        help="Version 2/2.5/3/...")
    parser.add_argument("--parameters", type=str, default="default",
                        help="Parameter count 3b, 4b, 7B, 8B, ...")
    parser.add_argument("--output_path", type=str, default="Outputs",
                        help="File path to store inference outputs CSV.")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    check_pytorch_gpu()
    main(args)
