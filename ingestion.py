import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import ReadTheDocsLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from streamlit import secrets

load_dotenv()

openai_api_key = secrets["OPENAI_API_KEY"]
pinecone_api_key = secrets["PINECONE_API_KEY"]
pinecone_index_name = secrets["PINECONE_INDEX_NAME"]

embeddings = OpenAIEmbeddings(api_key=openai_api_key, model="text-embedding-ada-002")


def ingest_docs():
    loader = ReadTheDocsLoader("./langchain-docs/api.python.langchain.com/en/latest", encoding='utf-8')
    raw_docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
    docs = text_splitter.split_documents(raw_docs)
    for doc in docs:
        new_url = doc.metadata["source"]
        new_url = new_url.replace("langchain-docs","https:/").replace("\\", "/")
        doc.metadata.update({"source": new_url})

    print(f"Loaded {len(docs)} documents. Storing to Pinecone...")
    PineconeVectorStore.from_documents(
        documents=docs, 
        embedding=embeddings,
        index_name=pinecone_index_name, 
        pinecone_api_key=pinecone_api_key
    )

if __name__ == "__main__":
    ingest_docs()
    print("Finished!")