from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from typing import List


def create_faiss_index(texts:List[str]):
    model_name = "sentence-transformers/all-mpnet-base-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)

    return FAISS.from_texts(texts=texts,embedding=embeddings)



def retrieve_relevant_docs(vectorstore: FAISS, query: str, k: int = 3):
    return vectorstore.similarity_search(query=query, k=k)
 