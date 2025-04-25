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

def generate_caption(labels_path):
    df = pd.read_csv(labels_path)

    dl = []
    failed = []
    invalid_image_count = 0

    batch = []

    for index, row in tqdm(df.iterrows()):
        image, tumor, tumor_type, image_type = row['image'], row['tumor'], row['tumor_type'], row['image_type']

        image_path = "/home/parthpk/NeuroChat/brain tumor/images_down"
        img_path = f"{image_path}/{tumor_type}/{image}"
        base64_image = encode_image(img_path)

        prompt_eval = f"Image metadata information:\n\n Tumor Presense: {tumor}, Tumor Type: {tumor_type}, Image view type: {image_type}"

        system_prompt = "You are an expert brain neurologist. Given a brain MRI image and its view type (axial, coronal, or sagittal), analyze the scan and determine if a brain tumor is present. If a tumor is present, describe the tumor type, location, appearance, and its effect on surrounding brain tissue. If no tumor is present, clearly and confidently explain that the scan is normal and provide visual reasoning (e.g., preserved symmetry, absence of abnormal signals, or mass effect). \nWrite your response as a single, medically accurate paragraph in natural language. Do not include any section headers, bullet points, field names, markdown formatting, or metadata — only the descriptive paragraph focusing on clinical reasoning and visual findings."


        PROMPT_MESSAGES = [
            {
                "role": "user",
                "content": [
                    {"type": "text","text": prompt_eval},
                    {"type": "image_url","image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}]
            },
            {"role": "system", "content": system_prompt}
        ]

        bat = {
            "custom_id": f"{row['no.']}",
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


    batc_output_path = "/home/ashmal/Courses/MedImgComputing/NeuroChat/Captions/1_brain_tumor/batchFiles"
    sub_batches = [batch[i:i + 1000] for i in range(0, len(batch), 1000)]
    os.makedirs(batc_output_path, exist_ok=True)

    for idx, sub_batch in enumerate(sub_batches):
        with open(f'{batc_output_path}/json_batch_{idx}.jsonl', 'w') as f:
            for entry in sub_batch:
                f.write(json.dumps(entry))
                f.write('\n')
    
    print(f"\n===========================\nDone making {len(sub_batches)} batches!")


        ## To check one by one response

        # try:
        #     response = client.chat.completions.create(
        #         model = deployment,
        #         messages = PROMPT_MESSAGES,
        #         max_tokens = 1000,
        #         temperature = 0.7
        #     )

        #     t_data = {
        #         "id": row["no."],
        #         "image": row["image"],
        #         "tumor_type": row["tumor_type"],
        #         "caption": response.choices[0].message.content,
        #     }

        #     dl.append(t_data)
        #     # print(dl[-1])
        #     # print()


        # except Exception as e:
        #     print(e)
        #     failed.append(row["no."])
        #     invalid_image_count += 1
        #     pass

        # if index == 5: 
        #     exit()




if __name__ == "__main__":
    labels_path = "/home/parthpk/NeuroChat/brain tumor/labels.csv"
    generate_caption(labels_path)
