
from modules.retriever import retrieve_documents
from modules.bm25_retriever import retrieve_bm25_documents

def retrieve_hybrid_documents(question: str):
    dense_documents = retrieve_documents(question)
    bm25_documents = retrieve_bm25_documents(question)
    
    fused_scores = {}
    document_map = {}
    
    def document_key(doc):
        return (
            doc.metadata.get("source", ""),
            doc.metadata.get("page", ""),
            doc.page_content
        )
        
        
    for rank, doc in enumerate(dense_documents, start=1):
        key = document_key(doc)

        document_map[key] = doc
        fused_scores[key] = fused_scores.get(key, 0) + 1 / (60 + rank)
        
        ranked_keys = sorted(
        fused_scores,
        key=fused_scores.get,
        reverse=True
    )
        
        ranked_documents = [
        document_map[key]
        for key in ranked_keys
    ]

    return ranked_documents