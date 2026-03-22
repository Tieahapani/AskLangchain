import os 
from dotenv import load_dotenv 
load_dotenv()

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from langsmith import Client, evaluate
from Rag.chain import ask 

client = Client()

STRATEGIES = ["similarity", "threshold", "mmr", "hybrid", "reranked"]



def run_strategy(strategy: str):
    """Creates a function that runs ask() with a specific strategy."""
    def predict(inputs: dict) -> dict:
        result = ask(inputs["question"], strategy=strategy)
        return {"answer": result["answer"]}
    return predict

def correctness_evaluator(run, example):
    """Check if the answer contains key information from the expected answer."""
    from langchain_google_genai import ChatGoogleGenerativeAI

    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
    
    prediction = run.outputs["answer"]
    reference = example.outputs["answer"]
    question = example.inputs["question"]

    grade_prompt = f"""You are grading a RAG system's answer.

Question: {question}
Expected Answer: {reference}
Actual Answer: {prediction}

Does the actual answer cover the key points from the expected answer? 
Grade as "correct" if it captures the main idea, even if worded differently.
Grade as "incorrect" if it misses key information or is wrong.

Respond with ONLY "correct" or "incorrect"."""

    response = llm.invoke(grade_prompt)
    score = 1 if "correct" in response.content.lower() else 0
    return {"key": "correctness", "score": score}

def faithfulness_evaluator(run, example): 
    """Check if the answer is grounded (not hallucinating)."""
    from langchain_google_genai import ChatGoogleGenerativeAI

    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
    
    prediction = run.outputs["answer"]
    question = example.inputs["question"]

    grade_prompt = f"""You are grading whether a RAG answer stays grounded.

Question: {question}
Answer: {prediction}

Does this answer appear to be grounded in documentation, or does it contain 
claims that seem fabricated or hallucinated? 

Respond with ONLY "grounded" or "hallucinated"."""

    response = llm.invoke(grade_prompt)
    score = 1 if "grounded" in response.content.lower() else 0
    return {"key": "faithfulness", "score": score}

if __name__ == "__main__": 
    dataset_name = "AskLangChain Eval"

    for strategy in STRATEGIES: 
        print(f"\n{'='*50}")
        print(f"Evaluating strategy: {strategy}")
        print(f"{'='*50}")

        results = evaluate(
            run_strategy(strategy), 
            data=dataset_name, 
            evaluators=[correctness_evaluator, faithfulness_evaluator], 
            experiment_prefix=f"asklangchain-{strategy}",
        )

        print(f"Results for {strategy}: {results}")

    print("\nDone! Check LangSmith dashboard to compare strategies.")





