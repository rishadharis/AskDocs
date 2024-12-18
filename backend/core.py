from dotenv import load_dotenv
from urllib.parse import urlparse
from pathlib import Path
import os
from langchain.chains.retrieval import create_retrieval_chain
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from streamlit import secrets

current_dir = Path(__file__).resolve().parent
dotenv_path = current_dir.parent / '.env'

load_dotenv(dotenv_path)

openai_api_key = secrets["OPENAI_API_KEY"]
pinecone_api_key = secrets["PINECONE_API_KEY"]
pinecone_index_name = secrets["PINECONE_INDEX_NAME"]
embeddings = OpenAIEmbeddings(api_key=openai_api_key, model="text-embedding-ada-002")

def run_llm(query: str):
    docsearch = PineconeVectorStore(index_name=pinecone_index_name, embedding=embeddings, pinecone_api_key=pinecone_api_key)
    llm = ChatOpenAI(api_key=openai_api_key, model="gpt-4o-mini", temperature=0, verbose=True)
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    stuff_documents_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)

    qa = create_retrieval_chain(
        retriever=docsearch.as_retriever(),
        combine_docs_chain=stuff_documents_chain
    )
    result = qa.invoke(input={"input": query})
    final_result = {
        "query": result["input"],
        "result": result["answer"],
        "source_documents" : result["context"]
    }
    return final_result
    

if __name__ == "__main__":
    result = run_llm("What is PromptTemplate and the very simple example of it?")
    print(result["source_documents"])
