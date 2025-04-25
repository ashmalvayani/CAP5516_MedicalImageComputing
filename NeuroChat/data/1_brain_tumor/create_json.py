import json
import csv

src = 'qs_test_bbtest.csv'
dest = 'NeuroChat_test_bbtest.json'
img_folder = '/home/parthpk/NeuroChat/'

lis = []
with open(src, 'r') as f:
    csvreader = csv.reader(f)
    next(csvreader, None)
    for line in csvreader:
        lis.append(line)
        
fin = []
for i in lis:
    fin.append({"id": i[1],"image": i[5],"conversations": [{"from": "human","value": f"<image>\n{i[6]}"},{"from": "gpt","value": f"{i[7]}"}]})

with open(dest, 'w') as f:
    json.dump(fin, f, indent=2)
