import chromadb
import os
import json
from openai import OpenAI

eval_set = [
        {"question":"How often do passwords have to change?","right_chunk":"chunk-0"},
        {"question":"What happens if a laptop fails its encryption check?","right_chunk":"chunk-1"},
        {"question":"Can I plug a USB stick into a finance computer?","right_chunk":"chunk-3"},
        {"question":"How long does guest wifi last?","right_chunk":"chunk-4"},
        {"question":"Where do I report a security incident?","right_chunk":"chunk-7"}
        ]


client = OpenAI(
        base_url = os.environ["OPENAI_BASE_URL"],
        api_key = os.environ["OPENAI_API_KEY"]
        )
chroma = chromadb.PersistentClient("/root/ragchroma")
collection = chroma.get_collection("itpolicy")

hit = 0
for data in eval_set:
    response=client.embeddings.create(
            model = "text-embedding-3-small",
            input = data["question"]
            )
    query=response.data[0].embedding

    results = collection.query(
            query_embeddings = query,
            n_results =2
            )
##results retruns ids,documents,distances,embeddings


    data['hit'] = True if data["right_chunk"] in results['ids'][0] else False
    hit += 1
    data['returned ids'] = results['ids'][0]

hit_rate = hit / 5
report = {"results":eval_set,"hit_rate":hit_rate}

with open("/root/rag-eval/retrieval-eval.json","w",encoding="utf-8") as f:
    json.dump(report,f,indent=4)

Output:
{
    "results": [
        {
            "question": "How often do passwords have to change?",
            "right_chunk": "chunk-0",
            "hit": true,
            "returned ids": [
                "chunk-0",
                "chunk-1"
            ]
        },
        {
            "question": "What happens if a laptop fails its encryption check?",
            "right_chunk": "chunk-1",
            "hit": true,
            "returned ids": [
                "chunk-1",
                "chunk-3"
            ]
        },
        {
            "question": "Can I plug a USB stick into a finance computer?",
            "right_chunk": "chunk-3",
            "hit": true,
            "returned ids": [
                "chunk-3",
                "chunk-5"
            ]
        },
        {
            "question": "How long does guest wifi last?",
            "right_chunk": "chunk-4",
            "hit": true,
            "returned ids": [
                "chunk-4",
                "chunk-1"
            ]
        },
        {
            "question": "Where do I report a security incident?",
            "right_chunk": "chunk-7",
            "hit": true,
            "returned ids": [
                "chunk-7",
                "chunk-0"
            ]
        }
    ],
    "hit_rate": 1.0
}
