import os
from pinecone import Pinecone
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv
load_dotenv()

def retrieve_documents(question: str):
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index = pc.Index(os.environ["PINECONE_INDEX_NAME"])

    embed_model = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        output_dimensionality=768
    )

    embedded_query = embed_model.embed_query(question)

    results = index.query(
        vector=embedded_query,
        top_k=10,
        include_metadata=True
    )

    documents = [
        Document(
            page_content=match["metadata"].get("text", ""),
            metadata=match["metadata"]
        )
        for match in results["matches"]
    ]

    return documents
