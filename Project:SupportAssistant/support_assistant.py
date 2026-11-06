from langchain_openai import OpenAIEmbeddings,ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

import glob
import sys

##
klist = []
kdata = ""
kfiles = glob.glob("/root/kb/*")
for kfilename in kfiles:
    with open(kfilename,"r") as k:
        klist.append(k.read())

kdata = " ".join(klist)
##
embedding = OpenAIEmbeddings(model="text-embedding-3-small")

splitter = RecursiveCharacterTextSplitter(
        chunk_size = 150,
        chunk_overlap = 30
        )
chunks = splitter.split_text(kdata)

chroma = Chroma.from_texts(
        embedding=embedding,
        collection_name="ops",
        texts=chunks,
        persist_directory="/root/chromadb"
        )

retriever = chroma.as_retriever(search_kwargs={'k':2})
##
model = ChatOpenAI(model = "glm-5.2")
##
prompt = ChatPromptTemplate.from_template(
        "You are Support Knowledge Assistant, you will answer user queries by strictly refering to 'Document' provided below only. If the information is not available in the document, you will say \"Information is not available in the documents\""
        "Document:{document}"
        "Question:{question}"
        )

def rag_chain(question):
    docs = retriever.invoke(question)
    response = "\n\n".join(doc.page_content for doc in docs)
    message = prompt.invoke({'document':response,'question':question})
    return(model.invoke(message).content)


if __name__ == '__main__':
    tfiles = glob.glob("/root/tickets/*")
    for tfilename in tfiles:
        tdata = []
        tstr = ""
        with open(tfilename,"r") as T:
            tdata.append(T.read())
        tstr = " ".join(tdata)
        response = rag_chain(tstr)
        replyfile = tfilename.replace("tickets","replies")
        with open(replyfile,"w") as w:
            w.write(response)


root@escbash-lab:~# 
