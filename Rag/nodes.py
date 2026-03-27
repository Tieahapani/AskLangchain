"""
Graph nodes for COT RAG + Self-Reflection pipeline.

Each node is a function that takes the current graph state,
performs one step, and returns updated state fields.
"""

from typing import TypedDict, List 
from langchain_core.documents import Document 
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from Rag.retriever import (
    get_similarity_retriever, get_threshold_retriever,
    get_mmr_retriever, get_hybrid_retriever, get_reranked_retriever,
)

### Shared State Schema 

class GraphState(TypedDict):
    question: str 
    strategy: str 
    documents: List[Document]
    filtered_documents: List[Document]
    generation: str 
    reasoning: str 
    reflection: str 
    is_acceptable: bool 
    retry_count: int 
    rewritten_question: str 

# ── LLM (reused across nodes) ────────────────────────────────────────

def get_llm():
    return ChatGoogleGenerativeAI(model="gemini-2.0-flash")


# ── Helper ────────────────────────────────────────────────────────────

def format_docs(docs: List[Document]) -> str:
    return "\n\n---\n\n".join(
        f"Source: {doc.metadata['source']}\nTitle: {doc.metadata['title']}\n\n{doc.page_content}"
        for doc in docs
    )


RETRIEVER_MAP = {
    "similarity": get_similarity_retriever,
    "threshold": get_threshold_retriever,
    "mmr": get_mmr_retriever,
    "hybrid": get_hybrid_retriever,
    "reranked": get_reranked_retriever,
}

## Node 1: Retrieve 

def retrieve(state: GraphState) -> dict: 
    """Fetch documents using the selected retrieval strategy."""
    strategy = state["strategy"]
    question = state.get("rewritten_question") or state["question"]

    retriever = RETRIEVER_MAP[strategy]()
    documents = retriever.invoke(question)

    return {"documents": documents}

## Node 2: Grade Documents 

GRADE_PROMPT = ChatPromptTemplate.from_template(
        """You are a relevance grader. Given a user question and a retrieved document,
decide if the document contains information relevant to answering the question.

Respond with ONLY "yes" or "no".

Question: {question}

Document content:
{document}

Is this document relevant?"""
)

def grade_documents(state: GraphState) -> dict: 
    """Grade each retrieved document for relevance. Keep only relevant ones."""
    question = state.get("rewritten_question") or state["question"]
    documents = state["documents"]
    llm = get_llm()
    chain = GRADE_PROMPT | llm | StrOutputParser()

    filtered = []
    for doc in documents: 
        result = chain.invoke({"question": question, "document": doc.page_content})
        if "yes" in result.strip().lower():
            filtered.append(doc)

    # if all docs were filtered out, keep originals to avoid empty context 
    if not filtered: 
        filtered =  documents 
    return {"filtered_documents": filtered}


## Node 3: Generate with Chain of Thought 

COT_PROMPT = ChatPromptTemplate.from_template(
    """You are a helpfule LangChain documentation assistant.Answer the user's question based ONLY on the following context from the LangChain docs.
    Context: 
    {context}
    
    Question: {question}
    
    Instructions: 
    1. First, write your REASONING — think step-by-step through the context,
   identifying which parts are relevant and how they connect to the question.
   Wrap your reasoning inside <reasoning>...</reasoning> tags.

2. Then, write your FINAL ANSWER — a clear, concise response.
   - Cite source URLs under a "Sources:" section.
   - Include code examples if relevant.
   - If the context doesn't contain enough information, say so honestly.

Begin:"""
)

def generate_cot(state: GraphState) -> dict: 
    """Generate an answer using Chain-of-Thought reasoning over filtered docs."""
    question = state.get("rewritten_question") or state["question"]
    docs = state["filtered_documents"]
    llm = get_llm()
    chain = COT_PROMPT | llm | StrOutputParser()

    raw_output = chain.invoke({
        "context": format_docs(docs), 
        "question": question, 
    })

     # Parse reasoning and answer from the output
    reasoning = ""
    answer = raw_output

    if "<reasoning>" in raw_output and "</reasoning>" in raw_output:
        start = raw_output.index("<reasoning>") + len("<reasoning>")
        end = raw_output.index("</reasoning>")
        reasoning = raw_output[start:end].strip()
        answer = raw_output[end + len("</reasoning>"):].strip()

    return {"reasoning": reasoning, "generation": answer}


## Node 4: Self-Reflect 

REFLECT_PROMPT = ChatPromptTemplate.from_template(
    """You are a quality evaluator for a RAG System. Given the question,the retrieved context, and the generated answer, evaluate the answer on two criteria:

1. **Grounded**: Is the answer fully supported by the provided context?
   (No hallucinated facts or claims beyond what the documents say.)

2. **Complete**: Does the answer actually address the user's question?

Context:
{context}

Question: {question}

Answer: {answer}

Respond in this exact format:
Grounded: yes/no
Complete: yes/no
Verdict: accept/retry
Reason: <one-line explanation> """
)

def self_reflection(state: GraphState) -> dict: 
    """Evaluate the generated answer for grounding and completeness."""
    question = state.get("rewritten_question") or state["question"]
    docs = state["filtered_documents"]
    answer = state["generation"]
    llm  = get_llm()
    chain = REFLECT_PROMPT | llm | StrOutputParser()

    reflection = chain.invoke({
        "context": format_docs(docs), 
        "question": question, 
        "answer": answer, 
    })

    is_acceptable = "accept" in reflection.lower()

    return {"reflection": reflection, "is_acceptable": is_acceptable}

# Node 5: Transform query (rewrite for better retrieval)


REWRITE_PROMPT = ChatPromptTemplate.from_template(
    """You are a query rewriter. The original question did not retrieve
good enough documents to produce a complete, grounded answer.

Original question: {question}

Self-reflection feedback:
{reflection}

Rewrite the question to be more specific and likely to retrieve
relevant LangChain documentation. Return ONLY the rewritten question, nothing else."""
)

def transform_query(state: GraphState) -> dict: 
    """Rewrite the question based on self-reflection feedback."""

    llm = get_llm()
    chain = REWRITE_PROMPT | llm | StrOutputParser()

    new_question = chain.invoke({
        "question": state["question"],
        "reflection": state["reflection"],
    })

    return {
        "rewritten_question": new_question.strip(),
        "retry_count": state.get("retry_count", 0) + 1,
    }


## Conditional Edge: Should we retry or accept? 
MAX_RETRIES = 2 

def should_retry(state: GraphState) -> str: 
    """Routing function for the conditional edge after self_reflect."""
    if state["is_acceptable"]: 
        return "accept"
    if state.get("retry_count", 0) >= MAX_RETRIES: 
        return "accept" # Give up after max retries, return best effort 
    
    return "retry"






