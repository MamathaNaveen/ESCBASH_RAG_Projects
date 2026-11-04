import os
import chromadb
from openai import OpenAI

openai_client = OpenAI(
    base_url=os.environ["OPENAI_BASE_URL"],
    api_key=os.environ["OPENAI_API_KEY"],
)

docs = [
    "Deploys to production happen every Tuesday at 4pm.",
    "Database backups run nightly at 2am and are kept for 30 days.",
    "On-call handover is Friday at 10am.",
]

response = openai_client.embeddings.create(
    model="text-embedding-3-small",
    input=docs,
)
vectors = [d.embedding for d in response.data]

chroma = chromadb.PersistentClient(path="/root/chroma-demo")
collection = chroma.get_or_create_collection("runbook")

collection.add(
    ids=["doc-0", "doc-1", "doc-2"],
    documents=docs,
    embeddings=vectors,
)

print("stored:", collection.count())
