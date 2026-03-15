import streamlit as st 
import os 
import sys 

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Load secrets for Streamlit Cloud deployment
if hasattr(st, "secrets") and "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Rag.chain import ask 

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

# Chat input
if question := st.chat_input("Ask about LangChain..."):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Get answer
    with st.chat_message("assistant"):
        with st.spinner("Searching docs..."):
            result = ask(question, strategy=strategy)

        st.markdown(result["answer"])

        # Show retrieved chunks
        docs_data = []
        with st.expander(f"Retrieved {len(result['docs'])} chunks"):
            for doc in result["docs"]:
                st.markdown(f"**{doc.metadata['title']}**")
                st.markdown(f"[{doc.metadata['source']}]({doc.metadata['source']})")
                st.code(doc.page_content[:300], language=None)
                st.divider()
                docs_data.append({
                    "title": doc.metadata["title"],
                    "source": doc.metadata["source"],
                    "text": doc.page_content[:300],
                })

    # Save to chat history
    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "docs": docs_data,
    })