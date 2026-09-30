from flashrank import Ranker, RerankRequest
from langchain_core.documents import Document

ranker = Ranker()

def rerank_documents(question: str, documents: list[Document], top_k: int = 3):
    
    passages = [
    {
        "id": i,
        "text": doc.page_content,
        "meta": doc.metadata
    }
    
  
    for i, doc in enumerate(documents)
]

    rerank_request = RerankRequest(
        query=question,
        passages=passages
    )
    
    results = ranker.rerank(rerank_request)
    
    reranked_documents = [
    documents[result["id"]]
    for result in results[:top_k]
]

    return reranked_documents

