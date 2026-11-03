
import json
import math
import os
from openai import OpenAI

client = OpenAI(base_url=os.environ["OPENAI_BASE_URL"],
                api_key=os.environ["OPENAI_API_KEY"])

def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    length_a = math.sqrt(sum(x * x for x in a))
    length_b = math.sqrt(sum(y * y for y in b))
    return dot / (length_a * length_b)

query = "I cannot log in to my account"
candidates = [
    "I am locked out and my password does not work",
    "How do I download my invoice history?",
    "The mobile app crashes when I open the camera",
    "Can I change the email address on my profile?",
    "Your latest update looks really nice",
]

response = client.embeddings.create(
    model="text-embedding-3-small",
    input=[query] + candidates,
)

query_vec = response.data[0].embedding
candidate_vecs = [d.embedding for d in response.data[1:]]

scored = [{"text": t, "score": round(cosine(query_vec, v), 3)}
          for t, v in zip(candidates, candidate_vecs)]
scored.sort(key=lambda item: item["score"], reverse=True)

json.dump(scored, open("/root/search/ranked.json", "w"), indent=2)
print(json.dumps(scored, indent=2))
