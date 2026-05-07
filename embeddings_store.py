from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from loader import load_and_chunk_pdf
import os
import time
import random
from dotenv import load_dotenv

load_dotenv()

# --- Retry / backoff settings ---
MAX_RETRIES = 5          # max attempts per batch
BASE_WAIT   = 10         # seconds to wait on first retry
MAX_WAIT    = 120        # cap per retry (seconds)

def _embed_with_retry(embeddings_model, chunks):
    """
    Build a FAISS store from chunks, retrying on 429 / RESOURCE_EXHAUSTED errors
    with exponential backoff + jitter.
    """
    # Split into small batches so a single 429 doesn't lose all work
    BATCH_SIZE = 5
    all_docs = []
    store = None

    for batch_start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[batch_start: batch_start + BATCH_SIZE]
        attempt = 0

        while True:
            try:
                if store is None:
                    store = FAISS.from_documents(documents=batch, embedding=embeddings_model)
                else:
                    batch_store = FAISS.from_documents(documents=batch, embedding=embeddings_model)
                    store.merge_from(batch_store)

                print(f"  ✓ Embedded batch {batch_start // BATCH_SIZE + 1} "
                      f"({batch_start + 1}–{batch_start + len(batch)}) / {len(chunks)} chunks")
                break  # success — move to next batch

            except Exception as e:
                err_str = str(e)
                is_rate_limit = ("429" in err_str or
                                 "RESOURCE_EXHAUSTED" in err_str or
                                 "quota" in err_str.lower())

                if is_rate_limit and attempt < MAX_RETRIES:
                    attempt += 1
                    # Exponential backoff with jitter
                    wait = min(BASE_WAIT * (2 ** (attempt - 1)) + random.uniform(0, 5), MAX_WAIT)
                    print(f"  ⚠ Rate limited (attempt {attempt}/{MAX_RETRIES}). "
                          f"Retrying in {wait:.1f}s …")
                    time.sleep(wait)
                else:
                    # Non-retryable error or retries exhausted
                    raise

    return store


def create_embeddings_and_store(pdf_path, store_name="faiss_index"):
    chunks, _ = load_and_chunk_pdf(pdf_path)
    print(f"Loaded {len(chunks)} chunks from PDF")

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    print("Embeddings model initialized — starting embedding with retry logic …")

    vector_store = _embed_with_retry(embeddings, chunks)

    print(f"Vector store created with {len(chunks)} embeddings")

    vector_store.save_local(store_name)
    print(f"Vector store saved as '{store_name}'")

    return vector_store


def load_existing_store(store_name="faiss_index"):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    vector_store = FAISS.load_local(
        store_name,
        embeddings,
        allow_dangerous_deserialization=True
    )
    return vector_store


if __name__ == "__main__":
    pdf_path = "sample_docs/EJ1200789.pdf"

    vector_store = create_embeddings_and_store(pdf_path)

    query = "What is the main topic?"
    results = vector_store.similarity_search(query, k=3)

    print(f"\n Search results for: '{query}'")
    for i, result in enumerate(results, 1):
        safe_content = result.page_content[:200].encode("ascii", "ignore").decode("ascii")
        print(f"\n{i}. {safe_content}...")