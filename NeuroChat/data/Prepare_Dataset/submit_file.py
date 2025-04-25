import openai
import json
import os
from openai import AzureOpenAI
from tqdm import tqdm

endpoint = os.getenv("ENDPOINT_URL", "")  
deployment = os.getenv("DEPLOYMENT_NAME", "gpt-4o-mini")
subscription_key = os.getenv("AZURE_OPENAI_API_KEY", "")

# Initialize Azure OpenAI Service client with key-based authentication    
client = AzureOpenAI(  
    azure_endpoint=endpoint,  
    api_key=subscription_key,  
    api_version="2024-10-21",
)

# Upload a file with a purpose of "batch"

folder_path = '/home/ashmal/Courses/MedImgComputing/NeuroChat/Captions/1_brain_tumor/batchFiles'
files = os.listdir(folder_path)

files = [file_name for file_name in files]
files.sort()

files = [os.path.join(folder_path, file) for file in files]

file_ids = []
for file_path in tqdm(files):
  file = client.files.create(
    file=open(file_path, "rb"), 
    purpose="batch"
  )
#   print(file.model_dump_json(indent=2))
  file_id = file.id

  file_ids.append(file.model_dump_json(indent=2))

with open("file_ids.json", 'w') as f:
    json.dump(file_ids, f, indent=2)
