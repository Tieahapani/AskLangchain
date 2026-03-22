# AskLangChain

A RAG-powered Q&A application that lets you ask natural language questions about the LangChain documentation and get grounded answers with source citations.

Instead of browsing through hundreds of LangChain doc pages, just ask a question and get a clear answer with links to the exact documentation pages.

## Features

- **5 Retrieval Strategies** — Compare how different retrieval methods affect answer quality:
  - **Similarity Search** — Returns the most similar chunks
  - **Score Threshold** — Only returns chunks above a relevance score
  - **MMR (Maximal Marginal Relevance)** — Balances relevance with diversity
  - **Hybrid (BM25 + Vector)** — Combines keyword and semantic search
  - **Hybrid + Reranking** — Hybrid search with cross-encoder reranking for best quality
- **Source Citations** — Every answer includes links back to the original doc pages
- **Chunk Transparency** — Expandable view showing exactly which chunks were retrieved
- **Chat Interface** — Conversational UI built with Streamlit

## Tech Stack

- **LangChain** — Document loading, chunking, retrieval, and RAG chain
- **FAISS** — Vector store for fast similarity search
- **Google Gemini** — Embeddings (gemini-embedding-001) and LLM (gemini-2.0-flash)
- **BM25** — Keyword-based retrieval for hybrid search
- **Cross-Encoder** — HuggingFace cross-encoder for reranking
- **Streamlit** — Web UI
- **BeautifulSoup** — Documentation scraping

## Architecture

User Query → Retrieval Strategy → FAISS/BM25 → Retrieved Chunks → Gemini LLM → Answer + Sources



1. **Ingestion (one-time):** Scrapes 100 LangChain doc pages, chunks them, embeds with Gemini, saves FAISS index
2. **Retrieval (runtime):** User picks a strategy, app retrieves relevant chunks from the index
3. **Generation (runtime):** Chunks are passed as context to Gemini, which generates an answer with source citations

## Project Structure

AskLangChain/
├── app.py                   # Streamlit UI
├── Rag/
│   ├── retriever.py         # 5 retrieval strategies
│   └── chain.py             # Prompt + Gemini answer generation
├── Scripts/
│   └── ingest_docs.py       # One-time: scrape, chunk, embed, save FAISS
├── VectorStore/             # Persisted FAISS index (not in repo)
├── requirements.txt
├── .env.example
└── .gitignore



## Setup

1. **Clone the repo:**
   ```bash
   git clone https://github.com/Tieahapani/AskLangChain.git
   cd AskLangChain
Install dependencies:


pip install -r requirements.txt
Set up environment variables:


cp .env.example .env
Add your Gemini API key to .env:


GOOGLE_API_KEY=your_gemini_api_key_here
Build the vector store (one-time):


python3 Scripts/ingest_docs.py
This scrapes LangChain docs, chunks them, embeds with Gemini, and saves the FAISS index to VectorStore/.

Run the app:


streamlit run app.py
Retrieval Strategies Explained
Strategy	How it works	Best for
Similarity	Returns k nearest vectors	General questions
Threshold	Only returns chunks above a score cutoff	When precision matters
MMR	Picks diverse chunks, avoids redundancy	Broad topic questions
Hybrid	BM25 keywords + vector semantics combined	Exact API names + concepts
Hybrid + Reranking	Hybrid → cross-encoder rescoring	Highest quality answers

## Here is the streamlit url that I have deployed, check it out! 
** https://asklangchain-beo4m5dnjb6qrtah4kterp.streamlit.app/** 
