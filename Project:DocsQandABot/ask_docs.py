
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
import glob
import json

###Read files
files = glob.glob("/root/docs/*")
datalist = []
for filename in files:
    with open(filename,"r") as f:
        datalist.append(f.read())
data = " ".join(datalist)    

###Text splitter
splitter = RecursiveCharacterTextSplitter(
        chunk_size = 200,
        chunk_overlap = 30
        )
chunks=splitter.split_text(data)

###Create embedding object
embeddings = OpenAIEmbeddings(model = "text-embedding-3-small")


###cerate chroma object
store = Chroma.from_texts(
        persist_directory="/root/docsbot/chroma",
        collection_name="askbot",
        embedding=embeddings,
        texts = chunks
        )
retriever = store.as_retriever(search_kwargs={'k':2})

###ChatOpenAI object
model = ChatOpenAI(model = "glm-5.2")

###prompt
prompt = ChatPromptTemplate.from_template(
        "You are Q&A Bot, You will answer user queries refering to the documents provided below. If the answer does not contain in the document you will say \"The information is not available in the document\""
        "Document:{context}"
        "Question:{question}"
        )


### RagChain
def ragchain(question):
    ##Steps 1 get relevent docs from retriever
    docslist=retriever.invoke(question)
    docs="\n\n".join(doc.page_content for doc in docslist)
    #Step2 query user question
    messages=prompt.invoke({"context":docs,"question":question})
    print(model.invoke(messages).content)
    


import sys
if __name__ == '__main__':
    ragchain(sys.argv[1])




