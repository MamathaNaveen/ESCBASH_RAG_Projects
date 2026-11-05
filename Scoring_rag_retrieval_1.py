import os
import chromadb
from openai import OpenAI

client = OpenAI(
    base_url=os.environ["OPENAI_BASE_URL"],
    api_key=os.environ["OPENAI_API_KEY"],
)

chroma = chromadb.PersistentClient(path="/root/rag-chroma-demo")
collection = chroma.get_collection("ops")

def rag_answer(question):
    q_vec = client.embeddings.create(
        model="text-embedding-3-small", input=question
    ).data[0].embedding
    top = collection.query(query_embeddings=[q_vec], n_results=2)["documents"][0]
    docs_text = "\n\n".join(f"Document {i+1}: {c}" for i, c in enumerate(top))
    prompt = ("Answer the question using only the documents below. If the answer "
              "is not in the documents, say \"The documents do not contain this "
              "information.\"\n\n"
              f"{docs_text}\n\nQuestion: {question}")
    reply = client.chat.completions.create(
        model="glm-5.2",
        messages=[{"role": "user", "content": prompt}],
    )
    return reply.choices[0].message.content

eval_set = [
    {"question": "How often do we back up the database?", "must_contain": "2am"},
    {"question": "When does new code reach production?", "must_contain": "Tuesday"},
    {"question": "What is the database password?", "must_contain": "not contain"},
]

correct = 0
for test in eval_set:
    answer = rag_answer(test["question"])
    ok = test["must_contain"].lower() in answer.lower()
    if ok:
        correct += 1
    print(f"{test['question'][:42]:45} {'PASS' if ok else 'FAIL'}")

print(f"answer accuracy: {correct}/{len(eval_set)} = {correct / len(eval_set)}")
