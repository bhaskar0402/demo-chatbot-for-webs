from dotenv import load_dotenv
import os
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

from ecommbot.data_converter import dataconveter

load_dotenv()

os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")

def ingestdata(status):

    embeddings=OpenAIEmbeddings()

    storage=status
    
    if storage==None:
        docs=dataconveter()
        vectors=Chroma.from_documents(docs, embeddings)
    else:
        return vectors
    return vectors

if __name__=='__main__':
    vectors =ingestdata(None)
    results = vectors.similarity_search("can you tell me the low budget sound basshead.")
    for res in results:
            print(f"* {res.page_content} [{res.metadata}]")
            

   