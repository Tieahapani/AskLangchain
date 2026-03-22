import os
from dotenv import load_dotenv
load_dotenv()
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_classic.retrievers import ContextualCompressionRetriever

def load_vectorstore(path: str = "VectorStore"): 
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)

def get_all_docs(path: str = "VectorStore"): 
    vs = load_vectorstore(path)
    return list(vs.docstore._dict.values())

## With the help of Similarity 
def get_similarity_retriever(k: int = 4):
    vs = load_vectorstore()
    return vs.as_retriever(search_type="similarity", search_kwargs={"k": k})

## With the help of threshold 

def get_threshold_retriever(score_threshold: float = 0.5, k : int = 4 ):
    vs = load_vectorstore()
    return vs.as_retriever(
        search_type="similarity_score_threshold", 
        search_kwargs={"score_threshold": score_threshold, "k": k}, 
    )

## With the help of MMR 

def get_mmr_retriever(k: int = 4, fetch_k: int = 20, lambda_mult: float = 0.5 ):
    vs = load_vectorstore()
    return vs.as_retriever(
        search_type="mmr", 
        search_kwargs={"k": k, "fetch_k": fetch_k, "lambda_mult": lambda_mult }, 
    )

## With the help of Hybrid search 

def get_hybrid_retriever(k: int = 4):
    vs = load_vectorstore()
    vector_retriever = vs.as_retriever(search_kwargs={"k": k})

    all_docs = get_all_docs()
    bm25_retriever = BM25Retriever.from_documents(all_docs, k=k)

    return EnsembleRetriever(
        retrievers = [bm25_retriever, vector_retriever],
        weights = [0.4, 0.6], 
    )

## With the help of hybrid and reranking 

def get_reranked_retriever(k: int = 4): 
    hybrid = get_hybrid_retriever(k=k*3)

    model = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    reranker = CrossEncoderReranker(model=model, top_n=k)

    return ContextualCompressionRetriever(
        base_compressor = reranker, 
        base_retriever = hybrid, 
    )

if __name__ == "__main__":
    retriever = get_similarity_retriever(k=3)
    docs = retriever.invoke("What is LangChain?")
    for doc in docs:
        print(f"\n--- {doc.metadata['title']} ---")
        print(doc.page_content[:200])
