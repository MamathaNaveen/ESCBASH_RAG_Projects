from langchain_openai import OpenAIEmbeddings,ChatOpenAI
import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
import sys

##Embedding Object
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

##create Chroma Object
chroma = Chroma(
        embedding_function=embeddings,
        persist_directory=("/root/lc-chroma")
        )

##Retriver
retriever = chroma.as_retriever(search_kwargs={"k":2})

##lLM model
model=ChatOpenAI(model="glm-5.2")

##Create prompt object
prompt = ChatPromptTemplate.from_template(
        "Answer the question using only the documents below. If the answer "
        "is not in the documents, say \"The documents do not contain this "
        "information.\"\n\n"
        "Documents:\n{context}\n\n"
        "Question: {question}"
        )

def rag_chain(question):
    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)
    messages = prompt.invoke({"context":context,"question":question})
    return model.invoke(messages).content

if  __name__ == "__main__":
    print(rag_chain(sys.argv[1]))
