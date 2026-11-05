from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter 
import json

with open("data.txt","r") as f:
    data=f.read()

splitter = RecursiveCharacterTextSplitter(
        chunk_size = 300,
        chunk_overlap = 50,
        )

chunks = splitter.split_text(data)

from langchain_openai import OpenAIEmbeddings


embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

store = Chroma.from_texts(
    texts=chunks,
    embedding=embeddings,
    persist_directory="/root/lc-chroma-demo",
)

retriever = store.as_retriever(search_kwargs={"k": 2})

results = retriever.invoke("How does a production rollout happen ?")
for doc in results:
    print(doc.page_content)
