
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import CharacterTextSplitter
import sys

data = []
with open("/root/runbook.md") as f:
    data.append(f.read())

para="\n\n".join(data)

##
embedding = OpenAIEmbeddings(model="text-embedding-3-small")
model = ChatOpenAI(model = "glm-5.2")
##
textsplitter = CharacterTextSplitter(
        separator="\n\n",
        chunk_overlap = 0,
        chunk_size = 300
        )
chunks = textsplitter.split_text(para)
##
chroma = Chroma.from_texts(
        embedding=embedding,
        persist_directory="/root/runbook-assistant/chroma",
        collection_name="ops",
        texts=chunks
        )
retriever = chroma.as_retriever(search_kwargs={'k':2})
##
prompt = ChatPromptTemplate.from_template(
        "You are runbook assistant that reads document provided below and answers user queries. You dont invent new procedures by yourself and answer to queries as its dangerous. It is strictly prohibited.It answer doesnt exist in document you will refuse to answer the query and Say \"The documents do not contain this information.\nHonest refusal:I refuse to answer this question\""
         "Document:{document}"
         "Question:{question}"
        )

def ragchain(question):
    docs=retriever.invoke(question)
    para="\n\n".join(doc.page_content for doc in docs)
    message=prompt.invoke({'document':para,'question':question})
    return(model.invoke(message).content)

if __name__ == '__main__':
    print(ragchain(sys.argv[1]))


