from modules.retriever import retrieve_documents
from modules.llm import get_llm_chain

def answer_question(question: str):
    documents = retrieve_documents(question)
    
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
