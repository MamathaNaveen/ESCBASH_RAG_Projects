from openai import OpenAI
import json
import os
import chromadb

query = "How far in advance do I ask for days off?"

#with open("/root/chroma-data.json","r",encoding="utf-8") as F:
#    data=json.load(F)

client = OpenAI(base_url=os.environ["OPENAI_BASE_URL"],
                api_key=os.environ["OPENAI_API_KEY"])
qc_embed=client.embeddings.create(
        model = "text-embedding-3-small",
        input=[query]
        )
qc_vec=qc_embed.data[0].embedding

chroma=chromadb.PersistentClient("/root/chroma-data")
collection=chroma.get_or_create_collection("handbook")

results = collection.query(
        query_embeddings = qc_vec,
        n_results = 2
        )
li = []
for text, distance in zip(results["documents"][0],results["distances"][0]):
    li.append(text)
jsontext = json.dumps(li)
print(jsontext)
with open("/root/chroma-results.json","w") as f:
    f.write(jsontext)
