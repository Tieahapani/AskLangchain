# AskLangChain

**LangChain docs are 100+ pages scattered across concepts, integrations, tutorials, and API references.** Developers waste time browsing instead of building. AskLangChain lets you ask in natural language and get grounded answers with source citations — instantly.

**[Try it live](https://asklangchain-beo4m5dnjb6qrtah4kterp.streamlit.app/)**

## What Makes This Different

This isn't a wrapper around an LLM. It's a **full RAG pipeline built from scratch** — web scraping → chunking → embedding → vector store → retrieval → generation — with **5 retrieval strategies**, **LangSmith observability**, and an **automated evaluation pipeline**.

## Features

### 5 Retrieval Strategies
Compare how different retrieval methods affect answer quality — side by side:

| Strategy | How it works | Best for |
|----------|-------------|----------|
| **Similarity Search** | Returns the k nearest vectors | General questions |
| **Score Threshold** | Only returns chunks above a relevance cutoff | When precision matters — better no answer than a wrong one |
| **MMR** | Picks diverse chunks, avoids redundancy | Broad topic questions where you need multiple angles |
| **Hybrid (BM25 + Vector)** | Combines keyword matching + semantic search | Exact API names like `ChatOpenAI` + conceptual queries |
| **Hybrid + Reranking** | Hybrid search → cross-encoder rescoring | Highest quality answers at the cost of slight latency |

### LangSmith Observability
Every query is traced end-to-end with custom metadata:
- Which retrieval strategy was used
- How many chunks were retrieved
- Full prompt sent to the LLM
- Latency per step (retrieval vs generation)
- User feedback (thumbs up/down) tied to each trace

### Automated Eval Pipeline
- **15-example benchmark dataset** uploaded to LangSmith
- **LLM-as-judge evaluators** for correctness and faithfulness
- Run all retrieval strategies against the same questions and compare scores
- Identifies which strategy performs best on which type of question

### User Feedback Loop
Thumbs up/down buttons in the UI send feedback directly to LangSmith traces — connecting real user experience back to specific retrieval runs for continuous improvement.

### Source Citations & Chunk Transparency
- Every answer cites the exact documentation pages it was derived from
- Expandable view shows which chunks were retrieved, so you can see exactly what the LLM was working with

## Architecture

Ingestion (one-time):
Scrape 100 doc pages → Chunk (1000 chars, 200 overlap) → Embed with Gemini → Save FAISS index

Runtime:
User Query → Strategy Selection → FAISS/BM25 Retrieval → Retrieved Chunks → Gemini 2.0 Flash → Answer + Sources
↓
LangSmith Trace
(metadata + feedback)



## Tech Stack

| Component | Technology |
|-----------|-----------|
| Framework | LangChain |
| Vector Store | FAISS |
| Embeddings | Google Gemini (`gemini-embedding-001`) |
| LLM | Google Gemini (`gemini-2.0-flash`) |
| Keyword Search | BM25 (rank-bm25) |
| Reranking | HuggingFace Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) |
| Observability | LangSmith (tracing, eval, feedback) |
| Web Scraping | BeautifulSoup + requests |
| UI | Streamlit |

## Project Structure

AskLangChain/
├── app.py                      # Streamlit UI with chat, strategy selector, feedback
├── Rag/
│   ├── retriever.py            # 5 retrieval strategies (similarity, threshold, MMR, hybrid, reranked)
│   └── chain.py                # RAG chain with LangSmith tracing + metadata
├── Scripts/
│   ├── ingest_docs.py          # One-time: scrape, chunk, embed, save FAISS index
│   ├── create_dataset.py       # Creates eval dataset in LangSmith (15 Q&A pairs)
│   └── eval.py                 # Runs eval pipeline across strategies with LLM-as-judge
├── VectorStore/                # Persisted FAISS index
├── requirements.txt
├── .env.example
└── .gitignore



## Setup

1. **Clone the repo:**
   ```bash
   git clone https://github.com/Tieahapani/AskLangchain.git
   cd AskLangchain
Install dependencies:


pip install -r requirements.txt
Set up environment variables:


cp .env.example .env
Add your keys to .env:


GOOGLE_API_KEY=your_gemini_api_key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_api_key
LANGCHAIN_PROJECT=AskLangChain
Build the vector store (one-time):


python3 Scripts/ingest_docs.py
Run the app:


streamlit run app.py
(Optional) Run the eval pipeline:


python3 Scripts/create_dataset.py
python3 Scripts/eval.py
What I Learned Building This
How different retrieval strategies (similarity, MMR, hybrid) produce fundamentally different results for the same query
Why hybrid search matters — keyword matching catches exact API names that semantic search misses
Cross-encoder reranking significantly improves precision but adds latency tradeoff
LangSmith tracing makes debugging RAG failures 10x easier — you can see exactly where retrieval went wrong vs generation
Building an eval pipeline forces you to think about what "good" retrieval actually means

