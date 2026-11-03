import json
with open("./handbook.txt","r") as f:
    data=f.read()

listd=[p.strip() for p in data.split("\n\n") if p.strip()] 

json.dump(listd,open("./chunks.json","w"))

import json
import os
from openai import OpenAI

client = OpenAI(
    base_url=os.environ["OPENAI_BASE_URL"],
    api_key=os.environ["OPENAI_API_KEY"],
)

chunks = json.load(open("chunks.json"))

response = client.embeddings.create(
    model="text-embedding-3-small",
    input=chunks,
)

index = []
for chunk, data in zip(chunks, response.data):
    index.append({"text": chunk, "embedding": data.embedding})

json.dump(index, open("index.json", "w"))
print(f"indexed {len(index)} chunks")
    
