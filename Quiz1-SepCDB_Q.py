
root@escbash-lab:~/ragchroma# cat create_chromadb.py 
import chromadb
from openai import OpenAI
import os

chunks = "Passwords must be changed on the first Monday of every quarter.Password managers approved by IT are listed on the intranet.\nLaptops are checked for disk encryption every 90 days. A failed check locks the laptop out of the VPN until IT clears it.\nPhones used for work email need a screen lock and remote wipe turned on. Personal apps are not restricted.\nUSB drives are blocked on all finance machines. Other teams can request an exception through the helpdesk.\nGuest wifi access lasts 24 hours. New guest codes come from reception.\nSoftware installs on work machines go through the self-service portal. Anything not in the portal needs an IT ticket.\nScreens must lock automatically after 10 minutes of no activity.\nSecurity incidents are reported to security@company.example within one hour of discovery."
lchunks = chunks.splitlines()
print(lchunks)
client = OpenAI(
        base_url = os.environ["OPENAI_BASE_URL"],
        api_key = os.environ["OPENAI_API_KEY"])
response = client.embeddings.create(
        model = "text-embedding-3-small",
        input = lchunks
        )

chunks_embed = [embed.embedding for embed in response.data]

chroma = chromadb.PersistentClient("/root/ragchroma")
collection = chroma.get_or_create_collection("itpolicy")
collection.add(
        ids = [str(i) for i in range(len(lchunks))],
        embeddings = chunks_embed,
        documents = lchunks
        )

#print("Indexed Collection")




root@escbash-lab:~/ragchroma# cat ragchroma-answer.py 
import os
from openai import OpenAI
import chromadb
import sys
croma=chromadb.PersistentClient("/root/ragchroma")
collection=croma.get_collection("itpolicy")

client = OpenAI(
        base_url = os.environ["OPENAI_BASE_URL"],
        api_key = os.environ["OPENAI_API_KEY"]
        )

query = sys.argv[1]

response=client.embeddings.create(
        model = "text-embedding-3-small",
        input = [query]
        )
qembed=response.data[0].embedding

results=collection.query(
        query_embeddings=[qembed],
        n_results=2
        )

docs_text_from_chroma = "\n\n" . join(f"Document {index + 1}: {data}" for index,data in enumerate(results["documents"][0]))

prompt = (
        "Answer the question using the documents given below. if the answer doesnt exist in document, Say \"The documents does not contain this inforamtion.\"\n\n"
        f"{docs_text_from_chroma}\n\n"
        f"Question: {query}"
        )

answer=client.chat.completions.create(
        model = "glm-5.2",
        messages = [{"role":"user","content":prompt}]
        )

print(answer.choices[0].message.content)


ANSWERS:
root@escbash-lab:~/ragchroma-answers# ls
q1.txt  q2.txt  q3.txt
root@escbash-lab:~/ragchroma-answers# cat *
Passwords must be changed on the first Monday of every quarter.
Laptops are checked for disk encryption every 90 days. According to Document 1, a failed check locks the laptop out of the VPN until IT clears it.
The documents do not contain this information.

The provided documents only cover:
- **Document 1:** Password change requirements and approved password managers
- **Document 2:** Software installation procedures and IT tickets

Neither document mentions anything about booking meeting rooms.
root@escbash-lab:~/ragchroma-answers# 
