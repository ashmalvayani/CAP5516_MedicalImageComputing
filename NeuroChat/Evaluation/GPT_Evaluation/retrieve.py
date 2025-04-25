import json
import os
from openai import AzureOpenAI

endpoint = os.getenv("ENDPOINT_URL", "")  
deployment = os.getenv("DEPLOYMENT_NAME", "gpt-4o-mini")
subscription_key = os.getenv("AZURE_OPENAI_API_KEY", "")

# Initialize Azure OpenAI Service client with key-based authentication    
client = AzureOpenAI(  
    azure_endpoint=endpoint,  
    api_key=subscription_key,
    api_version="2024-08-01-preview",
)

with open('batch_ids.json', 'r') as file:
    data = json.load(file)

# batch_ids = [d['batch_id'] for d in data]

batch_data = data

# Loop through full batch records instead of just IDs
for batch_info in batch_data:
    batch_id = batch_info['batch_id']
    model_name = batch_info.get('model_name', 'unknown_model')

    batch_response = client.batches.retrieve(batch_id)
    output_file_id = batch_response.output_file_id or batch_response.error_file_id

    if output_file_id:
        file_response = client.files.content(output_file_id)
        raw_responses = file_response.text.strip().split('\n')  

        final_outputs = []
        for index, raw_response in enumerate(raw_responses):
            json_response = dict(json.loads(raw_response))
            custom_id = json_response["custom_id"]

            try:
                caption = json_response["response"]["body"]["choices"][0]["message"]["content"]
            except:
                print(f"Unable to fetch caption for {batch_id}, {index}.")
                continue

            final_outputs.append({
                "id": custom_id,
                "caption": caption,
                "subset": model_name.split('_')[-1],
                "model_name": model_name.split('_')[0]
            })

        print(f"\n====================\n{batch_id} fetched total samples: {len(final_outputs)}\n")


        with open(f"/home/ashmal/Courses/MedImgComputing/NeuroChat/Captions/Qwen2-VL-Finetune/Evaluation/GPT_Eval_Outputs/{model_name}.json", "w") as f:
            json.dump(final_outputs, f, indent=2)
