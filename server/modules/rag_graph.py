from typing import TypedDict, NotRequired
from langchain_core.documents import Document
from modules.hybrid_retriever import retrieve_hybrid_documents
from modules.reranker import rerank_documents
from langchain_groq import ChatGroq
from modules.llm import get_llm_chain
from langgraph.graph import StateGraph, START, END
import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

grading_prompt = PromptTemplate(
    input_variables=["question", "context"],
    template="""
You are a relevance grader for a biomedical literature assistant.

Determine whether the retrieved context contains information relevant
to answering the user's question.

Question:
{question}

Retrieved context:
{context}

Respond with only one word:
YES or NO
"""
)


grader_llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="openai/gpt-oss-120b",
    temperature=0
)

grader_chain = grading_prompt | grader_llm | StrOutputParser()


rewrite_prompt = PromptTemplate(
    input_variables=["question"],
    template="""
You are a query rewriter for a biomedical literature retrieval system.

Rewrite the user's question so that it is clearer and more effective
for retrieving relevant biomedical literature.

Preserve the original meaning.

Original question:
{question}

Return only the rewritten question.
"""
)


rewrite_llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="openai/gpt-oss-120b",
    temperature=0
)

rewrite_chain = rewrite_prompt | rewrite_llm | StrOutputParser()


class GraphState(TypedDict):
    original_question: str
    search_query: str

    documents: NotRequired[list[Document]]
    answer: NotRequired[str]
    context_relevant: NotRequired[bool]
    rewrite_count: NotRequired[int]

    
def retrieve_node(state: GraphState):
    question = state["search_query"]
    documents = retrieve_hybrid_documents(question)

    return {
        "documents": documents
    }
    
def rerank_node(state: GraphState):
    question = state["search_query"]
    documents = state["documents"]

    reranked_documents = rerank_documents(
        question,
        documents
    )

    return {
        "documents": reranked_documents
    }
    
    
def grade_context_node(state: GraphState):
    question = state["original_question"]
    documents = state["documents"]

    context = "\n\n".join(
        doc.page_content for doc in documents
    )

    grade = grader_chain.invoke({
        "question": question,
        "context": context
    })

    is_relevant = grade.strip().upper() == "YES"

    return {
        "context_relevant": is_relevant
    }
    
    
MAX_REWRITES = 1

def route_after_grading(state: GraphState):
    if state["context_relevant"]:
        return "generate"

    if state.get("rewrite_count", 0) >= MAX_REWRITES:
        return "fallback"

    return "rewrite"



def rewrite_node(state: GraphState):
    question = state["search_query"]

    rewritten_question = rewrite_chain.invoke({
        "question": question
    })

    return {
        "search_query": rewritten_question.strip(),
        "rewrite_count": state.get("rewrite_count", 0) + 1
    }
    
     
    
def generate_node(state: GraphState):
    question = state["original_question"]
    documents = state["documents"]

    chain = get_llm_chain()

    result = chain.invoke({
        "query": question,
        "documents": documents
    })

    return {
        "answer": result["result"]
    }
    
def fallback_node(state: GraphState):
    return {
        "answer": (
            "I could not find enough relevant information "
            "in the uploaded biomedical literature to answer this question reliably."
        )
    }
    
       
    
# Create the graph
workflow = StateGraph(GraphState)

# Register nodes
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("rerank", rerank_node)
workflow.add_node("grade_context", grade_context_node)
workflow.add_node("rewrite", rewrite_node)
workflow.add_node("generate", generate_node)
workflow.add_node("fallback", fallback_node)

# Define the normal execution path
workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "rerank")
workflow.add_edge("rerank", "grade_context")

# Conditional decision after grading
workflow.add_conditional_edges(
    "grade_context",
    route_after_grading,
    {
        "generate": "generate",
        "rewrite": "rewrite",
        "fallback": "fallback"
    }
)

# A rewritten query goes through retrieval again
workflow.add_edge("rewrite", "retrieve")

# Both possible final paths terminate the graph
workflow.add_edge("generate", END)
workflow.add_edge("fallback", END)

# Compile into an executable graph
rag_graph = workflow.compile()