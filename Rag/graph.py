"""LangGraph state graph that writes the COT RAG + Self-Reflection nodes into a cyclic workflow with conditonal retry logic"""

from langgraph.graph import StateGraph, END 

from Rag.nodes import (
    GraphState, 
    retrieve, 
    grade_documents, 
    generate_cot, 
    self_reflection, 
    transform_query, 
    should_retry, 
)

def build_graph():
    """Build and compile the COT RAG + Self-Reflection graph."""
    graph = StateGraph(GraphState)

    ## Add Nodes 
    graph.add_node("retrieve", retrieve)
    graph.add_node("grade_documents", grade_documents)
    graph.add_node("generate_cot", generate_cot)
    graph.add_node("self_reflect", self_reflection)
    graph.add_node("transform_query", transform_query)

    ## Set Entry Point 

    graph.set_entry_point("retrieve")

    # Linear edges 

    graph.add_edge("retrieve", "grade_documents")
    graph.add_edge("grade_documents", "generate_cot")
    graph.add_edge("generate_cot", "self_reflect")

    ## Conditonal edge: accept or retry 

    graph.add_conditional_edges(
        "self_reflect", 
        should_retry, 
        {
            "accept": END, 
            "retry": "transform_query", 
        }, 
    )

    # After rewritten loop back to retrieve 
    graph.add_edge("transform_query", "retrieve")

    return graph.compile()

# Pre-built instance for easy import 
cot_rag_graph = build_graph()

