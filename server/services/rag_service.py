from modules.hybrid_retriever import retrieve_hybrid_documents
from modules.reranker import rerank_documents
from modules.llm import get_llm_chain

def answer_question(question: str):
    documents = retrieve_hybrid_documents(question)
    documents = rerank_documents(question, documents)
    chain = get_llm_chain()

    result = chain.invoke({
        "query": question,
        "documents": documents
    })

    return {
    "response": result["result"],
    "sources": [
        doc.metadata.get("source", "")
        for doc in documents
    ]
}
