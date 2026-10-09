from modules.rag_graph import rag_graph


def answer_question(question: str):

    result = rag_graph.invoke({
        "original_question": question,
        "search_query": question
    })

    documents = (
    result.get("documents", [])
    if result.get("context_relevant", False)
    else [])

    return {
        "response": result.get("answer", ""),
        "sources": [
            doc.metadata.get("source", "")
            for doc in documents
        ]
    }