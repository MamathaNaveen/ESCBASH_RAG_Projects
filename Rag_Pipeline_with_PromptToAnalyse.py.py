#Chroma or Consine gives the best match from source provided, but it nevers checks if answer is right.
##hence feed the highest matched document to prompt and let prompt analyse if matched document is relevent to query. 

import math
import os
from openai import OpenAI

client = OpenAI(
    base_url=os.environ["OPENAI_BASE_URL"],
    api_key=os.environ["OPENAI_API_KEY"],
)

def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    length_a = math.sqrt(sum(x * x for x in a))
    length_b = math.sqrt(sum(y * y for y in b))
    return dot / (length_a * length_b)

# phase 1: index (runs once)
chunks = [
    "Deploys to production happen every Tuesday at 4pm.",
    "Database backups run nightly at 2am and are kept for 30 days.",
    "If the site goes down, check the load balancer health page first.",
]
response = client.embeddings.create(model="text-embedding-3-small", input=chunks)
vectors = [d.embedding for d in response.data]

# phase 2: retrieve (runs per question)
question = "How often do we back up the database?"
q_response = client.embeddings.create(model="text-embedding-3-small", input=question)
q_vec = q_response.data[0].embedding

best = max(zip(chunks, vectors), key=lambda pair: cosine(q_vec, pair[1]))[0]
print("retrieved:", best)

# phase 3: generate (grounded prompt, both rules)
prompt = (
    "Answer the question using only the document below. If the answer "
    "is not in the document, say \"The document does not contain this "
    "information.\"\n\n"
    f"Document: {best}\n\n"
    f"Question: {question}"
)
answer = client.chat.completions.create(
    model="glm-5.2",
    messages=[{"role": "user", "content": prompt}],
)
print("answer:", answer.choices[0].message.content)
