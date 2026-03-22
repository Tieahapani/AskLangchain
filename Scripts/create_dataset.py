import os 
from dotenv import load_dotenv 
load_dotenv()

from langsmith import Client 

client = Client()

dataset_name = "AskLangChain Eval"
dataset = client.create_dataset(dataset_name, description="Q&A pairs for evaluating AskLangChain retrieval strategies")

examples = [
    {
        "question": "How do I install LangChain?",
        "answer": "Install LangChain with pip install langchain. For specific integrations, install additional packages like langchain-openai or langchain-google-genai."
    },
    {
        "question": "What is a vector store in LangChain?",
        "answer": "A vector store is a database that stores embeddings and allows you to search by semantic similarity. LangChain supports FAISS, Chroma, Pinecone, and others."
    },
    {
        "question": "How do I use ChatOpenAI?",
        "answer": "Import ChatOpenAI from langchain_openai, provide your API key, and call invoke with a message. It wraps OpenAI's chat models."
    },
    {
        "question": "What is retrieval augmented generation?",
        "answer": "RAG retrieves relevant documents from a knowledge base and passes them as context to an LLM to generate grounded answers."
    },
    {
        "question": "How do I set up FAISS with LangChain?",
        "answer": "Install faiss-cpu, create embeddings, then use FAISS.from_documents() or FAISS.from_texts() to build the index."
    },
    {
        "question": "What is an agent in LangChain?",
        "answer": "An agent uses an LLM to decide which tools to call and in what order to complete a task. LangChain provides prebuilt agents and tools."
    },
    {
        "question": "How do I use document loaders?",
        "answer": "Document loaders load data from various sources like PDFs, web pages, and databases into LangChain Document objects."
    },
    {
        "question": "What is LangGraph?",
        "answer": "LangGraph is a framework for building stateful, multi-step agent workflows as graphs with nodes and edges."
    },
    {
        "question": "How do I add memory to a LangChain agent?",
        "answer": "Use short-term memory for conversation history within a session, or long-term memory to persist information across sessions using checkpointers."
    },
    {
        "question": "What embedding models does LangChain support?",
        "answer": "LangChain supports OpenAI, Google Gemini, HuggingFace, Cohere, AWS Bedrock, and many other embedding model providers."
    },
    {
        "question": "How do I use text splitters?",
        "answer": "Use RecursiveCharacterTextSplitter to split documents into chunks with configurable chunk_size and chunk_overlap parameters."
    },
    {
        "question": "What is structured output in LangChain?",
        "answer": "Structured output lets you force LLMs to return responses in a specific format like JSON, using Pydantic models or output parsers."
    },
    {
        "question": "How do I use Anthropic Claude with LangChain?",
        "answer": "Install langchain-anthropic, import ChatAnthropic, provide your API key, and use it like any other chat model."
    },
    {
        "question": "What is the Model Context Protocol?",
        "answer": "MCP is a protocol that allows LLMs to connect to external tools and data sources in a standardized way."
    },
    {
        "question": "How do I deploy a LangChain application?",
        "answer": "Use LangSmith for deployment and hosting, or deploy as a standard Python web app with FastAPI or Streamlit."
    },
]

for ex in examples:
    client.create_example(
        inputs={"question": ex["question"]},
        outputs={"answer": ex["answer"]},
        dataset_id=dataset.id,
    )





