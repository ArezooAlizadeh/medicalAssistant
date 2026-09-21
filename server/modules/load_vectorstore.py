import os
import time
from pathlib import Path
from dotenv import load_dotenv
from tqdm.auto import tqdm
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from google.genai.errors import ClientError

# Google's free-tier embedding quota is a hard 100 requests/minute, metered
# per individual text embedded (not per API call) - a single batch call
# embedding 90 texts counts as 90 against the quota. We track how many texts
# we've submitted in the last 60s and pause before submitting more than a
# safety margin under that cap.
EMBED_BATCH_SIZE = 25
MAX_TEXTS_PER_MINUTE = 90
RATE_LIMIT_RETRY_SECONDS = 60
MAX_RATE_LIMIT_RETRIES = 5

_request_log = []  # list of (timestamp, text_count)


def _wait_for_rate_limit_capacity(batch_size):
    now = time.monotonic()
    _request_log[:] = [(t, c) for t, c in _request_log if now - t < 60]
    while sum(c for _, c in _request_log) + batch_size > MAX_TEXTS_PER_MINUTE:
        sleep_time = 60 - (now - _request_log[0][0]) + 1
        print(f"⏳ Approaching rate limit, waiting {sleep_time:.0f}s...")
        time.sleep(max(sleep_time, 0))
        now = time.monotonic()
        _request_log[:] = [(t, c) for t, c in _request_log if now - t < 60]


def _embed_documents_rate_limited(embed_model, texts):
    embeddings = []
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i:i + EMBED_BATCH_SIZE]
        for attempt in range(MAX_RATE_LIMIT_RETRIES):
            _wait_for_rate_limit_capacity(len(batch))
            try:
                embeddings.extend(embed_model.embed_documents(batch))
                _request_log.append((time.monotonic(), len(batch)))
                break
            except ClientError as e:
                if e.code == 429 and attempt < MAX_RATE_LIMIT_RETRIES - 1:
                    print(f"⏳ Rate limited, waiting {RATE_LIMIT_RETRY_SECONDS}s before retrying batch...")
                    time.sleep(RATE_LIMIT_RETRY_SECONDS)
                else:
                    raise
    return embeddings

load_dotenv()

GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY=os.getenv("PINECONE_API_KEY")
PINECONE_ENV="us-east-1"
PINECONE_INDEX_NAME="medicalindex"

os.environ["GOOGLE_API_KEY"]=GOOGLE_API_KEY

UPLOAD_DIR="./uploaded_docs"
os.makedirs(UPLOAD_DIR,exist_ok=True)


# initialize pinecone instance
pc=Pinecone(api_key=PINECONE_API_KEY)
spec=ServerlessSpec(cloud="aws",region=PINECONE_ENV)
existing_indexes=[i["name"] for i in pc.list_indexes()]


if PINECONE_INDEX_NAME not in existing_indexes:
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=768,
        metric="dotproduct",
        spec=spec
    )
    while not pc.describe_index(PINECONE_INDEX_NAME).status["ready"]:
        time.sleep(1)


index=pc.Index(PINECONE_INDEX_NAME)

# load,split,embed and upsert pdf docs content

def load_vectorstore(uploaded_files):
    embed_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", output_dimensionality=768)
    file_paths = []

    for file in uploaded_files:
        save_path = Path(UPLOAD_DIR) / file.filename
        with open(save_path, "wb") as f:
            f.write(file.file.read())
        file_paths.append(str(save_path))

    for file_path in file_paths:
        loader = PyPDFLoader(file_path)
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(documents)

        texts = [chunk.page_content for chunk in chunks]
        metadatas = [{**chunk.metadata, "text": chunk.page_content} for chunk in chunks]
        ids = [f"{Path(file_path).stem}-{i}" for i in range(len(chunks))]

        print(f"🔍 Embedding {len(texts)} chunks...")
        embeddings = _embed_documents_rate_limited(embed_model, texts)

        print("📤 Uploading to Pinecone...")
        with tqdm(total=len(embeddings), desc="Upserting to Pinecone") as progress:
            index.upsert(vectors=zip(ids, embeddings, metadatas))
            progress.update(len(embeddings))

        print(f"✅ Upload complete for {file_path}")
