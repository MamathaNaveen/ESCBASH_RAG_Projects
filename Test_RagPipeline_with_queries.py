
Code:
import os
import math
from openai import OpenAI

handbook = ["Release notes are published on the team wiki every Friday afternoon.",
            "The test suite must pass before any branch is merged to main.",
            "Production alerts page the on-call engineer through the paging app.",
            "New services need a security review before their first deploy."]

query = ["Where do the release notes get published?",
         "What is the wifi password in the office?"
         ]

client = OpenAI(
        base_url = os.environ["OPENAI_BASE_URL"],
        api_key = os.environ["OPENAI_API_KEY"]
        )

response = client.embeddings.create(
        model = "text-embedding-3-small",
        input = handbook
        )

responseq = client.embeddings.create(
        model = "text-embedding-3-small",
        input = query
        )

hb_vec = [embed.embedding for embed in response.data]
query_vec = [embed.embedding for embed in responseq.data]

def cosine(a,b):
    p = sum((x * y) for x,y in zip(a,b))
    q = math.sqrt(sum((x*x) for x in a))
    r = math.sqrt(sum((y*y) for y in b))
    cosine = p / ( q + r)
    return(cosine)

hb_q1_cosine = [cosine(vec,query_vec[0]) for vec in hb_vec]
hb_q2_cosine = [cosine(vec,query_vec[1]) for vec in hb_vec]
#print(hb_q2_cosine)


max_index_q1 = int(hb_q1_cosine.index(max(hb_q1_cosine)))
max_index_q2 = int(hb_q2_cosine.index(max(hb_q2_cosine)))

#print(max_index_q2)

q1_ans = handbook[max_index_q1]
q2_ans = handbook[max_index_q2]
print(q2_ans)
prompt1 = (
        "Answer the question using only the document below. if the answer "
        "is not in the document, say \"the docuemtn does not contain this information. \n\n"
        f"Document: {query[0]}"
        f"Answer: {q1_ans}"
        )

prompt2 = (
        "Answer the question using only the document below. if the answer "
        "is not in the document, say \"the docuemtn does not contain this information. \n\n"
        f"Document: {query[1]}"
        f"Answer: {q2_ans}"
        )

res1 = client.chat.completions.create(
        model = "glm-5.2",
        messages = [{"role":"user","content":prompt1}]
        )
res2 = client.chat.completions.create(
        model = "glm-5.2",
        messages = [{"role":"user","content":prompt2}]
        )

with open("/root/pipeline/retrieved.txt","w") as f:
    f.write(q1_ans)
    
with open("/root/pipeline/answer-1.txt","w") as f:
    f.write(res1.choices[0].message.content)
with open("/root/pipeline/answer-2.txt","w") as f:
    f.write(res2.choices[0].message.content)
    

print(res1)
print(res2)


OUTPUT:
root@escbash-lab:~/pipeline# cat answer-2.txt 
The document does not contain this information.root@escbash-lab:~/pipeline# 
root@escbash-lab:~/pipeline# cat answer-1.txt 
Release notes are published on the team wiki every Friday afternoon.
root@escbash-lab:~/pipeline# cat retrieved.txt 
Release notes are published on the team wiki every Friday afternoon.root@escbash-lab:~/pipeline# cat ^C






root@escbash-lab:~/pipeline# 
