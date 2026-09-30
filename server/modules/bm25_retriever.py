import json
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document

BM25_DATA_PATH = "./bm25_chunks.json"

def load_bm25_chunks():
    with open(BM25_DATA_PATH, "r", encoding="utf-8") as f:
        stored_chunks = json.load(f)

    documents = [
        Document(
            page_content=item["text"],
            metadata=item["metadata"]
        )
        for item in stored_chunks
    ]
    return documents


def tokenize_documents(documents):
    tokenized_corpus = [
        doc.page_content.lower().split()
        for doc in documents
    ]

    return tokenized_corpus

def build_bm25_index(documents):
    tokenized_corpus = tokenize_documents(documents)
    bm25 = BM25Okapi(tokenized_corpus)

    return bm25


def retrieve_bm25_documents(question: str, top_k: int = 10):
    documents = load_bm25_chunks()
    bm25 = build_bm25_index(documents)

    tokenized_query = question.lower().split()
    scores = bm25.get_scores(tokenized_query)
    
    ranked_indices = sorted(
    range(len(scores)),
    key=lambda i: scores[i],
    reverse=True
    )[:top_k]
    
    ranked_documents = [
    documents[i]
    for i in ranked_indices
]

    return ranked_documents
    