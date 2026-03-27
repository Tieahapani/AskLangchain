import streamlit as st
import os
import sys
import uuid
from langsmith import Client as LangSmithClient

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Load secrets for Streamlit Cloud deployment
if hasattr(st, "secrets"):
    for key in ["GOOGLE_API_KEY", "LANGCHAIN_TRACING_V2", "LANGCHAIN_API_KEY", "LANGCHAIN_PROJECT"]:
        if key in st.secrets:
            os.environ[key] = st.secrets[key]

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Rag.chain import ask, ask_cot_reflect 

# Page config
st.set_page_config(page_title="AskLangChain", page_icon="🔗", layout="wide")

# Sidebar
with st.sidebar:
    st.title("AskLangChain")
    st.markdown("Ask anything about the LangChain docs")
    st.divider()

    strategy = st.selectbox(
        "Retrieval Strategy",
        ["similarity", "threshold", "mmr", "hybrid", "reranked"],
        format_func=lambda x: {
            "similarity": "Similarity Search",
            "threshold": "Score Threshold",
            "mmr": "MMR (Diverse Results)",
            "hybrid": "Hybrid (BM25 + Vector)",
            "reranked": "Hybrid + Reranking",
        }[x],
    )

    st.caption({
        "similarity": "Returns the k most similar chunks.",
        "threshold": "Only returns chunks above a relevance score.",
        "mmr": "Balances relevance with diversity.",
        "hybrid": "Combines keyword + semantic search.",
        "reranked": "Hybrid search with cross-encoder reranking.",
    }[strategy])

    st.divider()
    mode = st.radio(
        "RAG Mode",
        ["standard", "cot_reflect"],
        format_func=lambda x: {
            "standard": "Standard RAG",
            "cot_reflect": "COT + Self-Reflection",
        }[x],
    )
    st.caption({
        "standard": "Single-pass retrieval and generation.",
        "cot_reflect": "Chain-of-Thought reasoning with self-reflection and retry loop.",
    }[mode])

    st.divider()
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "docs" in msg:
            with st.expander(f"Retrieved {len(msg['docs'])} chunks"):
                for doc in msg["docs"]:
                    st.markdown(f"**{doc['title']}**")
                    st.markdown(f"[{doc['source']}]({doc['source']})")
                    st.code(doc["text"][:300], language=None)
                    st.divider()
        if msg.get("run_id") and msg["role"] == "assistant":
            col1, col2, _ = st.columns([1, 1, 10])
            with col1:
                if st.button("👍", key=f"hist_up_{msg['run_id']}"):
                    ls_client = LangSmithClient()
                    ls_client.create_feedback(uuid.UUID(msg["run_id"]), key="user_score", score=1)
                    st.toast("Thanks for the feedback!")
            with col2:
                if st.button("👎", key=f"hist_down_{msg['run_id']}"):
                    ls_client = LangSmithClient()
                    ls_client.create_feedback(uuid.UUID(msg["run_id"]), key="user_score", score=0)
                    st.toast("Thanks — we'll improve!")

# Chat input
if question := st.chat_input("Ask about LangChain..."):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Get answer
    with st.chat_message("assistant"):
        if mode == "cot_reflect":
            with st.spinner("Thinking step-by-step..."):
                result = ask_cot_reflect(question, strategy=strategy)
        else:
            with st.spinner("Searching docs..."):
                result = ask(question, strategy=strategy)

        st.markdown(result["answer"])

        # COT extras: reasoning, reflection, retry info
        if mode == "cot_reflect":
            if result.get("reasoning"):
                with st.expander("Reasoning Trace (Chain-of-Thought)"):
                    st.markdown(result["reasoning"])

            if result.get("reflection"):
                with st.expander("Self-Reflection"):
                    st.markdown(result["reflection"])

            if result.get("retry_count", 0) > 0:
                st.info(f"Query was rewritten {result['retry_count']} time(s). Final query: _{result.get('rewritten_question', '')}_")

        # Show retrieved chunks
        docs_data = []
        docs_list = result.get("docs", [])
        with st.expander(f"Retrieved {len(docs_list)} chunks"):
            for doc in docs_list:
                st.markdown(f"**{doc.metadata['title']}**")
                st.markdown(f"[{doc.metadata['source']}]({doc.metadata['source']})")
                st.code(doc.page_content[:300], language=None)
                st.divider()
                docs_data.append({
                    "title": doc.metadata["title"],
                    "source": doc.metadata["source"],
                    "text": doc.page_content[:300],
                })

        if result.get("run_id"):
             col1, col2, _ = st.columns([1, 1, 10])
             with col1:
                if st.button("👍", key=f"up_{result['run_id']}"):
                    ls_client = LangSmithClient()
                    ls_client.create_feedback(uuid.UUID(result["run_id"]), key="user_score", score=1)
                    st.toast("Thanks for the feedback!")
             with col2:
                if st.button("👎", key=f"down_{result['run_id']}"):
                    ls_client = LangSmithClient()
                    ls_client.create_feedback(uuid.UUID(result["run_id"]), key="user_score", score=0)
                    st.toast("Thanks — we'll improve!")

    # Save to chat history
    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "docs": docs_data,
        "run_id": result.get("run_id"),
    })