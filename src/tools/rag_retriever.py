import hashlib
import statistics
import time

from functools import lru_cache
from pathlib import Path
from typing import Any

import chromadb

from chromadb.errors import NotFoundError

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from transformers import AutoTokenizer


# ==================================================
# 1. PROJECT CONFIGURATION
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DOCUMENT_DIR = PROJECT_ROOT / "data" / "processed"
CHROMA_DIR = PROJECT_ROOT / "data" / "chroma_db"

COLLECTION_NAME = "sprints_project_knowledge"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# The embedding model has a maximum sequence length of 256.
# We reserve space for special tokens.
CHUNK_SIZE = 240
CHUNK_OVERLAP = 50

MAX_EMBEDDING_SEQUENCE_LENGTH = 256

DEFAULT_RETRIEVAL_K = 4


# ==================================================
# 2. KNOWLEDGE SOURCE CONFIGURATION
# ==================================================

KNOWLEDGE_SOURCES = [
    {
        "filename": "project_brief.md",
        "source_type": "project_brief",
        "trust_level": "high",
        "limitations": (
            "Primary written source for project requirements."
        ),
    },
    {
        "filename": "client_kickoff_deck.md",
        "source_type": "kickoff_deck",
        "trust_level": "medium",
        "limitations": (
            "Some requirements on slides 3 and 4 are truncated "
            "in the original presentation."
        ),
    },
    {
        "filename": "client_discovery_call.md",
        "source_type": "discovery_call_transcript",
        "trust_level": "medium",
        "limitations": (
            "AI-generated transcript. Human verification "
            "against the original audio is still required."
        ),
    },
]


# ==================================================
# 3. MODEL INITIALIZATION
# ==================================================

@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Initialize and cache the Hugging Face embedding model.

    The model is loaded once and reused across retrieval calls.
    CPU execution is selected for compatibility.
    """

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={
            "device": "cpu"
        },
        encode_kwargs={
            "normalize_embeddings": True
        },
    )


@lru_cache(maxsize=1)
def get_tokenizer():
    """
    Load and cache the tokenizer associated with the
    selected Sentence Transformer model.

    The tokenizer is used for chunk measurement only.
    Its maximum length is increased to prevent warnings
    when measuring complete documents before splitting.

    This does NOT change the embedding model's actual
    maximum input sequence length.
    """

    tokenizer = AutoTokenizer.from_pretrained(
        EMBEDDING_MODEL
    )

    tokenizer.model_max_length = 10**9

    return tokenizer


def count_tokens(
    text: str,
    add_special_tokens: bool = False,
) -> int:
    """
    Count tokens using the embedding model tokenizer.

    Special tokens are excluded during chunk splitting.
    They are included during embedding-limit validation.
    """

    tokenizer = get_tokenizer()

    return len(
        tokenizer.encode(
            text,
            add_special_tokens=add_special_tokens,
        )
    )


# ==================================================
# 4. TEXT SPLITTING
# ==================================================

def get_text_splitter() -> RecursiveCharacterTextSplitter:
    """
    Create a token-aware recursive text splitter.

    Configuration:
        Maximum chunk size: 240 tokens.
        Overlap: 50 tokens.

    The chunk size leaves room for special tokens
    within the embedding model's 256-token limit.
    """

    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=count_tokens,
        separators=[
            "\n## ",
            "\n### ",
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ],
        keep_separator=True,
    )


# ==================================================
# 5. DOCUMENT LOADING
# ==================================================

def load_knowledge_documents() -> list[Document]:
    """
    Load the approved project knowledge documents.

    Each document receives source metadata and a trust level.

    SQLite databases and audio binaries are intentionally
    excluded from this RAG document-loading process.
    """

    documents = []

    for source in KNOWLEDGE_SOURCES:

        file_path = DOCUMENT_DIR / source["filename"]
        if not file_path.exists():
            fixture_path = PROJECT_ROOT / "tests" / "fixtures" / "curriculum_knowledge" / source["filename"]
            if fixture_path.exists():
                file_path = fixture_path
            else:
                raise FileNotFoundError(
                    f"Required knowledge source not found: {file_path}"
                )

        content = file_path.read_text(
            encoding="utf-8"
        ).strip()

        if not content:
            raise ValueError(
                f"Knowledge source is empty: {file_path.name}"
            )

        metadata = {
            "source": file_path.name,
            "source_type": source["source_type"],
            "trust_level": source["trust_level"],
            "limitations": source["limitations"],
            "source_path": str(
                file_path.relative_to(PROJECT_ROOT)
            ),
        }

        documents.append(
            Document(
                page_content=content,
                metadata=metadata,
            )
        )

    return documents


# ==================================================
# 6. DOCUMENT CHUNKING
# ==================================================

def split_documents(
    documents: list[Document],
) -> list[Document]:
    """
    Split source documents into token-based chunks.

    Each chunk preserves its original source metadata.

    Chunk indexes restart for every source document.
    """

    splitter = get_text_splitter()

    chunks = []

    for document in documents:

        source_chunks = splitter.split_documents(
            [document]
        )

        for index, chunk in enumerate(source_chunks):

            chunk.metadata["chunk_index"] = index

            chunks.append(chunk)

    if not chunks:
        raise ValueError(
            "Document splitting generated no chunks."
        )

    return chunks


# ==================================================
# 7. CHUNK VALIDATION
# ==================================================

def validate_chunks(
    chunks: list[Document],
) -> dict[str, Any]:
    """
    Validate chunk sizes against the embedding model limit.

    Each chunk must fit within the actual model sequence
    limit, including special tokens.
    """

    token_counts = []

    for index, chunk in enumerate(chunks):

        token_count = count_tokens(
            chunk.page_content,
            add_special_tokens=False,
        )

        sequence_length = count_tokens(
            chunk.page_content,
            add_special_tokens=True,
        )

        token_counts.append(token_count)

        if token_count > CHUNK_SIZE:
            raise ValueError(
                f"Chunk {index} exceeds configured chunk size: "
                f"{token_count} > {CHUNK_SIZE}"
            )

        if sequence_length > MAX_EMBEDDING_SEQUENCE_LENGTH:
            raise ValueError(
                f"Chunk {index} exceeds embedding model limit: "
                f"{sequence_length} > "
                f"{MAX_EMBEDDING_SEQUENCE_LENGTH}"
            )

    return {
        "total_chunks": len(chunks),
        "maximum_tokens": max(token_counts),
        "minimum_tokens": min(token_counts),
        "average_tokens": round(
            statistics.mean(token_counts),
            2,
        ),
        "embedding_limit": MAX_EMBEDDING_SEQUENCE_LENGTH,
        "validation_status": "passed",
    }


# ==================================================
# 8. DETERMINISTIC CHUNK IDENTIFIERS
# ==================================================

def generate_chunk_id(
    chunk: Document,
) -> str:
    """
    Generate a deterministic identifier for a document chunk.

    The identifier includes its source, index, and content hash.
    """

    source = chunk.metadata["source"]

    chunk_index = chunk.metadata["chunk_index"]

    content_hash = hashlib.sha256(
        chunk.page_content.encode("utf-8")
    ).hexdigest()

    unique_value = (
        f"{source}:{chunk_index}:{content_hash}"
    )

    return hashlib.sha256(
        unique_value.encode("utf-8")
    ).hexdigest()


# ==================================================
# 9. CHROMADB COLLECTION MANAGEMENT
# ==================================================

def reset_collection() -> None:
    """
    Delete the previous ChromaDB collection.

    This ensures a clean rebuild and prevents old chunks
    from remaining after document changes.
    """

    CHROMA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR)
    )

    try:

        client.get_collection(
            name=COLLECTION_NAME
        )

        client.delete_collection(
            name=COLLECTION_NAME
        )

        print("Previous ChromaDB collection deleted.")

    except NotFoundError:

        print("No previous collection found.")

    get_vector_store.cache_clear()


@lru_cache(maxsize=1)
def get_vector_store() -> Chroma:
    """
    Connect to the persistent ChromaDB collection.

    The vector store instance is cached to avoid
    repeatedly initializing the same connection.
    """

    chroma_sqlite = CHROMA_DIR / "chroma.sqlite3"
    if not (CHROMA_DIR.exists() and chroma_sqlite.exists()):
        fixture_dir = PROJECT_ROOT / "tests" / "fixtures" / "curriculum_knowledge"
        if fixture_dir.exists():
            print("Auto-provisioning ChromaDB vector storage from curriculum fixtures...")
            ingest_documents()
        else:
            raise FileNotFoundError(
                "ChromaDB storage does not exist. "
                "Run document ingestion first."
            )

    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=str(CHROMA_DIR),
    )


# ==================================================
# 10. DOCUMENT INGESTION
# ==================================================

def ingest_documents() -> dict[str, Any]:
    """
    Build the complete RAG knowledge base.

    Pipeline:
        1. Load source documents.
        2. Split documents into token-based chunks.
        3. Validate all chunks.
        4. Initialize the embedding model.
        5. Verify embedding execution.
        6. Reset the previous collection.
        7. Store the new chunks in ChromaDB.
        8. Verify the final collection count.

    Returns:
        Ingestion summary.
    """

    print("=" * 60)
    print("SPRINTS RAG INGESTION")
    print("=" * 60)

    # Step 1: Load documents

    print("\n[1/7] Loading knowledge sources...")

    documents = load_knowledge_documents()

    print(f"Documents loaded: {len(documents)}")

    # Step 2: Split documents

    print("\n[2/7] Splitting documents...")

    chunks = split_documents(documents)

    print(f"Chunks generated: {len(chunks)}")

    # Step 3: Validate chunks

    print("\n[3/7] Validating chunk sizes...")

    validation = validate_chunks(chunks)

    print(f"Maximum tokens: {validation['maximum_tokens']}")
    print(f"Minimum tokens: {validation['minimum_tokens']}")
    print(f"Average tokens: {validation['average_tokens']}")

    print("Chunk validation: PASSED")

    # Step 4: Initialize embeddings

    print("\n[4/7] Initializing embedding model...")

    embeddings = get_embeddings()

    # Step 5: Verify embedding execution

    print("\n[5/7] Testing embedding generation...")

    test_embedding = embeddings.embed_query(
        "Sprints project knowledge base test."
    )

    if not test_embedding:
        raise RuntimeError(
            "Embedding model returned an empty vector."
        )

    print(
        f"Embedding dimensions: {len(test_embedding)}"
    )

    # Step 6: Reset previous collection

    print("\n[6/7] Preparing ChromaDB collection...")

    reset_collection()

    # Step 7: Store documents

    print("\n[7/7] Storing documents in ChromaDB...")

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )

    ids = [
        generate_chunk_id(chunk)
        for chunk in chunks
    ]

    if len(ids) != len(set(ids)):
        raise ValueError(
            "Duplicate chunk identifiers detected."
        )

    vector_store.add_documents(
        documents=chunks,
        ids=ids,
    )

    stored_count = vector_store._collection.count()

    if stored_count != len(chunks):
        raise RuntimeError(
            f"Storage verification failed: "
            f"{stored_count} stored vs {len(chunks)} generated."
        )

    result = {
        "status": "success",
        "documents_loaded": len(documents),
        "chunks_created": len(chunks),
        "chunks_stored": stored_count,
        "maximum_chunk_tokens": validation["maximum_tokens"],
        "embedding_dimensions": len(test_embedding),
        "collection_name": COLLECTION_NAME,
        "persist_directory": str(CHROMA_DIR),
    }

    print("\n" + "=" * 60)
    print("INGESTION COMPLETED SUCCESSFULLY")
    print("=" * 60)

    for key, value in result.items():
        print(f"{key}: {value}")

    get_vector_store.cache_clear()

    return result


# ==================================================
# 11. CONTEXT RETRIEVAL
# ==================================================

def retrieve_context(
    query: str,
    k: int = DEFAULT_RETRIEVAL_K,
) -> list[Document]:
    """
    Retrieve the most relevant knowledge chunks.

    Args:
        query: Natural language question.
        k: Maximum number of chunks to retrieve.

    Returns:
        Retrieved documents with source metadata.
    """

    if not isinstance(query, str) or not query.strip():
        raise ValueError(
            "Search query must be a non-empty string."
        )

    if not isinstance(k, int) or k < 1:
        raise ValueError(
            "k must be a positive integer."
        )

    vector_store = get_vector_store()

    collection_count = vector_store._collection.count()

    if collection_count == 0:
        raise RuntimeError(
            "Knowledge base is empty. "
            "Run ingestion before retrieval."
        )

    return vector_store.similarity_search(
        query,
        k=min(k, collection_count),
    )


# ==================================================
# 12. RETRIEVAL EVALUATION
# ==================================================

RETRIEVAL_TEST_CASES = [
    {
        "query": (
            "What are the Sprint 1 document ingestion "
            "and vector database requirements?"
        ),
        "expected_source": "project_brief.md",
        "expected_terms": [
            "ChromaDB",
            "500-token",
        ],
    },
    {
        "query": (
            "Why must learner Python code execution "
            "be strictly sandboxed?"
        ),
        "expected_source": "client_discovery_call.md",
        "expected_terms": [
            "sandboxed",
            "security",
        ],
    },
    {
        "query": (
            "What is the target reduction in student "
            "waiting time?"
        ),
        "expected_source": "client_discovery_call.md",
        "expected_terms": [
            "80%",
        ],
    },
    {
        "query": (
            "What structured output fields are required "
            "for Python assignment grading?"
        ),
        "expected_source": "project_brief.md",
        "expected_terms": [
            "score",
            "feedback_points",
            "pass_status",
        ],
    },
]


def evaluate_retrieval(
    k: int = 5,
    repetitions: int = 3,
) -> dict[str, Any]:
    """
    Evaluate source retrieval and response latency.

    The evaluation checks:
        - Whether the expected source was retrieved.
        - Whether expected evidence terms appear in results.
        - Average, median, and p95 retrieval latency.

    This is a small regression test, not a complete
    information-retrieval benchmark.
    """

    if repetitions < 1:
        raise ValueError(
            "Repetitions must be at least 1."
        )

    print("\n" + "=" * 60)
    print("RAG RETRIEVAL EVALUATION")
    print("=" * 60)

    # Warm up model and vector store before measurement.

    retrieve_context(
        "Sprints project requirements",
        k=1,
    )

    evaluation_results = []
    all_latencies = []

    for test in RETRIEVAL_TEST_CASES:

        query = test["query"]

        latency_runs = []
        retrieved_documents = []

        for _ in range(repetitions):

            start_time = time.perf_counter()

            documents = retrieve_context(
                query,
                k=k,
            )

            elapsed_ms = (
                time.perf_counter() - start_time
            ) * 1000

            latency_runs.append(elapsed_ms)

            retrieved_documents = documents

        all_latencies.extend(latency_runs)

        retrieved_sources = [
            doc.metadata.get("source")
            for doc in retrieved_documents
        ]

        combined_content = "\n".join(
            doc.page_content.lower()
            for doc in retrieved_documents
        )

        expected_source_found = (
            test["expected_source"] in retrieved_sources
        )

        matched_terms = [
            term
            for term in test["expected_terms"]
            if term.lower() in combined_content
        ]

        evidence_found = (
            len(matched_terms) > 0
        )

        result = {
            "query": query,
            "expected_source": test["expected_source"],
            "retrieved_sources": retrieved_sources,
            "expected_source_found": expected_source_found,
            "matched_terms": matched_terms,
            "evidence_found": evidence_found,
            "average_latency_ms": round(
                statistics.mean(latency_runs),
                2,
            ),
        }

        evaluation_results.append(result)

        print("\nQuestion:", query)

        print(
            "Expected source:",
            test["expected_source"],
        )

        print(
            "Retrieved sources:",
            retrieved_sources,
        )

        print(
            "Source found:",
            expected_source_found,
        )

        print(
            "Evidence terms found:",
            matched_terms,
        )

        print(
            "Average latency:",
            result["average_latency_ms"],
            "ms",
        )

    source_matches = sum(
        result["expected_source_found"]
        for result in evaluation_results
    )

    evidence_matches = sum(
        result["evidence_found"]
        for result in evaluation_results
    )

    total_questions = len(evaluation_results)

    sorted_latencies = sorted(all_latencies)

    p95_index = min(
        len(sorted_latencies) - 1,
        int(0.95 * len(sorted_latencies)),
    )

    summary = {
        "total_questions": total_questions,
        "source_matches": source_matches,
        "source_match_rate": round(
            source_matches / total_questions,
            3,
        ),
        "evidence_matches": evidence_matches,
        "evidence_match_rate": round(
            evidence_matches / total_questions,
            3,
        ),
        "average_latency_ms": round(
            statistics.mean(all_latencies),
            2,
        ),
        "median_latency_ms": round(
            statistics.median(all_latencies),
            2,
        ),
        "p95_latency_ms": round(
            sorted_latencies[p95_index],
            2,
        ),
        "under_1000ms": all(
            latency < 1000
            for latency in all_latencies
        ),
    }

    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    for key, value in summary.items():
        print(f"{key}: {value}")

    return {
        "summary": summary,
        "details": evaluation_results,
    }


# ==================================================
# 13. MAIN
# ==================================================

if __name__ == "__main__":

    ingest_documents()