import os
import json
import base64
from PIL import Image
from io import BytesIO
import openai
from tqdm import tqdm
import pandas as pd

from openai import AzureOpenAI

endpoint = os.getenv("ENDPOINT_URL", "")  
deployment = os.getenv("DEPLOYMENT_NAME", "gpt-4o-mini")
subscription_key = os.getenv("AZURE_OPENAI_API_KEY", "")


# Initialize Azure OpenAI Service client with key-based authentication    
client = AzureOpenAI(  
    azure_endpoint=endpoint,  
    api_key=subscription_key,  
    api_version="2024-10-21",
)

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def generate_caption(results_path, model_name, subset):
    df = pd.read_csv(results_path)

    dl = []
    failed = []
    invalid_image_count = 0

    batch = []

    for index, row in tqdm(df.iterrows()):
        id, question_type, gt_question, gt_answer, model_output = row["id"], row["question_type"], row["gt_question"], row["gt_answer"], row["model_output"]

        try:
            # print(model_output)
            model_output = model_output.replace('```','').replace('json','')
            model_output = json.loads(model_output)["answer"]
        except Exception as e:
            model_output = model_output


        prompt_eval = f"Question: {gt_question}\nGrount Truth Answer: {gt_answer}\nPredicted Answer: {model_output}"

        if question_type == 'caption':
            system_prompt = f"""You are an expert medical evaluator.
            Evaluate the following ground truth answer and the predicted answer to a medical imaging question. Your evaluation should consider:
            - Accuracy: How correct is the information in the predicted answer?
            - Relevancy: How well does the predicted answer address the question?
            - Consistency: How consistent and thorough is the answer compared to the ground truth?
            - Insightfulness: Does the answer demonstrate the presence of tumor, its type, its location, and the image viewing angle (Sagittal/Coronal/Axial)?

            Give a single overall rating on a scale from 0 to 10, and reason for justification where:
            0 = completely incorrect, irrelevant, and unhelpful,
            10 = perfectly accurate, highly relevant, consistent, and insightful.

            Respond only with the numeric score (no explanation). Strictly respond in a json format as follows {{'score': SCORE_HERE, 'reason': REASON_HERE}}.
            """
        
        else:
            system_prompt = f"""You are an expert medical evaluator.
                Evaluate the following ground truth answer and the predicted answer to a medical imaging question. Your evaluation should consider:
                - Accuracy: How correct is the information in the predicted answer?
                - Relevancy: How well does the predicted answer address the question?

                Give a single overall rating on a scale from 0 to 10, and reason for justification where:
                0 = incorrect answer
                10 = correct answer compared to the ground truth

                Respond only with the numeric score (no explanation). Strictly respond in a json format as follows {{'score': SCORE_HERE, 'reason': REASON_HERE}}.
                """

        PROMPT_MESSAGES = [
            {
                "role": "user",
                "content": [{"type": "text","text": prompt_eval}],
            },
            {"role": "system", "content": system_prompt}
        ]

        bat = {
            "custom_id": id,
            "method": "POST", 
            "url": "/chat/completions", 
            "body": {
                    "model": deployment,
                    "messages": PROMPT_MESSAGES,
                    "max_tokens": 1000,
                    "temperature":0.7
                }
            }
        
        batch.append(bat)


    output_folder = f"/home/ashmal/Courses/MedImgComputing/NeuroChat/Captions/Qwen2-VL-Finetune/Evaluation/GPT_Eval/"
    output_file = f"{output_folder}/{model_name}_{subset}.jsonl"

    with open(f'{output_file}', 'w') as f:
        for idx, entry in enumerate(batch):
            f.write(json.dumps(entry))
            f.write('\n')
    
    print(f"\n===========================\nDone making batch for {model_name} model!")


        ## To check one by one response

    # try:
    #     response = client.chat.completions.create(
    #         model = deployment,
    #         messages = PROMPT_MESSAGES,
    #         max_tokens = 1000,
    #         temperature = 0.7
    #     )

    #     t_data = {
    #         "id": id,
    #         "question_type": question_type,
    #         "gt_question": gt_question,
    #         "gt_answer": gt_answer,
    #         "model_output": model_output,
    #         "score": response.choices[0].message.content,
    #     }

    #     dl.append(t_data)
    #     print(dl[-1])
    #     print()


    # except Exception as e:
    #     print(e)
    #     failed.append(row["no."])
    #     invalid_image_count += 1
    #     pass

    # if index == 5: 
    #     exit()




if __name__ == "__main__":
    # results_path = "/home/ashmal/Courses/MedImgComputing/NeuroChat/Captions/Qwen2-VL-Finetune/Evaluation/Outputs/baseline/btmri/tt_results_Qwen2-VL-7B-Instruct/results.csv"

    output_file_path = "/home/ashmal/Courses/MedImgComputing/NeuroChat/Captions/Qwen2-VL-Finetune/Evaluation/Outputs_GT"
    files = os.listdir(output_file_path)

    output_files = [os.path.join(output_file_path, file) for file in files]

    for output_file in output_files:
        print(output_file)
        
        split_file = output_file.split('/')[-1]
        model_name = split_file.split('_')[0]
        subset = split_file.split('_')[1]

        generate_caption(output_file, model_name, subset)
