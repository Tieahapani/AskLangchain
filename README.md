# AskLangChain

A smart Q&A assistant for LangChain documentation. Instead of digging through 100+ pages of docs, just ask a question in plain English and get an accurate answer with links to the source.

[Try it live](https://asklangchain-beo4m5dnjb6qrtah4kterp.streamlit.app/)

## The Problem

LangChain's documentation is spread across concepts, tutorials, integrations, and API references. Finding the right answer means opening multiple tabs and scanning through walls of text. AskLangChain solves this — ask a question, get an answer, see exactly where it came from.

## How It Works

1. **Scraped and indexed** 100 pages of LangChain documentation
2. **Split the content** into small, searchable chunks
3. **When you ask a question**, the system finds the most relevant chunks and generates an answer grounded in the actual docs
4. **Every answer includes source links** so you can verify and read further

## Key Features

### 5 Ways to Search

Not all search methods work equally well for every question. AskLangChain lets you switch between 5 different retrieval strategies to compare results:

| Strategy | What it does |
|----------|-------------|
| **Similarity Search** | Finds the closest matching chunks |
| **Score Threshold** | Only returns results above a quality cutoff |
| **MMR** | Picks diverse results to avoid repetition |
| **Hybrid Search** | Combines keyword search + meaning-based search |
| **Hybrid + Reranking** | Hybrid search with a second pass to pick the best results |

### Smart Answering with Step-by-Step Reasoning

The app has two modes:

- **Standard Mode** — retrieves relevant docs and generates an answer in one pass
- **COT + Self-Reflection Mode** — a more thorough pipeline that:
  1. Retrieves documents
  2. Filters out irrelevant ones before answering
  3. Thinks through the context step-by-step before writing a response
  4. Checks its own answer — is it actually supported by the docs? Does it answer the question?
  5. If the answer isn't good enough, it rewrites the question and tries again (up to 2 retries)

This is built using **LangGraph**, which lets you define multi-step workflows with loops and decision points.

In practice, the filtering + step-by-step reasoning produced well-grounded answers even on vague questions — self-reflection rarely needed to trigger retries because the earlier steps already cleaned up the context.

You can inspect the full reasoning and self-check in expandable sections in the UI.

### Full Tracing & Observability

Every question is tracked end-to-end using **LangSmith**:
- Which search strategy was used
- What chunks were retrieved
- How long each step took
- Whether the user gave a thumbs up or down

### Evaluation Pipeline

- 15 test questions with expected answers
- Automated scoring for correctness and faithfulness
- Compare all search strategies against the same questions to see which one performs best

### User Feedback

Thumbs up/down buttons on every answer feed directly into LangSmith traces — linking real user experience to specific system runs.

## Architecture

**One-time Setup:**
Scrape docs → Split into chunks → Create embeddings → Save to search index

**Standard Mode:**
Question → Search → Relevant Chunks → LLM → Answer with Sources

**COT + Self-Reflection Mode:**

| Step | What happens |
|------|-------------|
| 1 | Search for relevant document chunks |
| 2 | Filter out irrelevant chunks |
| 3 | Reason step-by-step through the context |
| 4 | Self-check: Is the answer grounded and complete? |
| 5a | If yes → return the answer |
| 5b | If no → rewrite the question and go back to step 1 (max 2 retries) |


**Project Structure section — replace with:**

```markdown
## Project Structure

AskLangChain/
│
├── app.py                      # Chat UI with strategy picker, mode toggle, feedback buttons
│
├── Rag/
│   ├── retriever.py            # 5 search strategies
│   ├── chain.py                # Standard + COT answer generation
│   ├── nodes.py                # Individual steps: retrieve, filter, reason, reflect, rewrite
│   └── graph.py                # Wires the steps into a workflow with retry logic
│
├── Scripts/
│   ├── ingest_docs.py          # Scrape and index documentation (run once)
│   ├── create_dataset.py       # Create test dataset in LangSmith
│   └── eval.py                 # Run evaluation across all strategies
│
├── VectorStore/                # Saved search index
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
Build the search index (one-time):


python3 Scripts/ingest_docs.py
Run the app:


streamlit run app.py
(Optional) Run evaluations:


python3 Scripts/create_dataset.py
python3 Scripts/eval.py

## What I Learned

- Different search methods give very different results for the same question — there's no single best approach
- Combining keyword search with meaning-based search catches things that either method misses alone
- Reranking results with a second model improves accuracy but adds a speed tradeoff
- Filtering irrelevant chunks before answering makes a bigger difference than expected — the model reasons much better with clean context
- Step-by-step reasoning over retrieved docs produces more reliable answers than asking the model to answer directly
- End-to-end tracing makes it easy to debug exactly where things went wrong
- Building a test suite forces you to define what a "good answer" actually looks like

