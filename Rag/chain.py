import os 
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

from Rag.retriever import (
    get_similarity_retriever, get_threshold_retriever, get_mmr_retriever, get_hybrid_retriever, get_reranked_retriever, 
)

load_dotenv()
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

PROMPT_TEMPLATE = """You are a helpful LangChain documentation assistant.
Answer the user's question based ONLY on the following context from the LangChain docs.
If the context doesn't contain enough information to answer, say so honestly.

Context: 
{context}

Question: {question}

Instructions: 
- Give a clear, concise answer 
- Cite the source URLs at the end of your answer under a "Sources:" section 
- If code examples are relevant, include them
"""

prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

def format_docs(docs):
    return "\n\n---\n\n".join(
        f"Source: {doc.metadata['source']}\nTitle: {doc.metadata['title']}\n\n{doc.page_content}"
        for doc in docs
    )

def get_retriever(strategy: str = "similarity"):
    strategies = {
        "similarity": get_similarity_retriever,
        "threshold": get_threshold_retriever,
        "mmr": get_mmr_retriever,
        "hybrid": get_hybrid_retriever,
        "reranked": get_reranked_retriever,
    }
    return strategies[strategy]()


def ask(question: str, strategy: str = "similarity"):
    retriever = get_retriever(strategy)
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

    # Get retrieved docs (to show in UI later)
    docs = retriever.invoke(question)

    # Build the chain
    chain = prompt | llm | StrOutputParser()

    # Run it
    answer = chain.invoke({
        "context": format_docs(docs),
        "question": question,
    })

    return {"answer": answer, "docs": docs}

if __name__ == "__main__":
    result = ask("How do I set up FAISS with LangChain?", strategy="similarity")
    print(result["answer"])
    print(f"\nRetrieved {len(result['docs'])} chunks")