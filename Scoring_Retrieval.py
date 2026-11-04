import os
import chromadb
from openai import OpenAI

client = OpenAI(
    base_url=os.environ["OPENAI_BASE_URL"],
    api_key=os.environ["OPENAI_API_KEY"],
)

chroma = chromadb.PersistentClient(path="/root/rag-chroma-demo")
collection = chroma.get_collection("ops")

eval_set = [
    {"question": "How often do we back up the database?", "right_chunk": "chunk-1"},
    {"question": "When does new code reach production?", "right_chunk": "chunk-0"},
    {"question": "What is the first check when the site is down?", "right_chunk": "chunk-2"},
    {"question": "Who takes over on-call duty, and when?", "right_chunk": "chunk-3"},
]

hits = 0
for test in eval_set:
    q_vec = client.embeddings.create(
        model="text-embedding-3-small", input=test["question"]
    ).data[0].embedding
    results = collection.query(query_embeddings=[q_vec], n_results=2)
    returned_ids = results["ids"][0]
    hit = test["right_chunk"] in returned_ids
    if hit:
        hits += 1
    print(f"{test['question'][:45]:48} {'HIT' if hit else 'MISS'} {returned_ids}")

print(f"hit rate: {hits}/{len(eval_set)} = {hits / len(eval_set)}")
