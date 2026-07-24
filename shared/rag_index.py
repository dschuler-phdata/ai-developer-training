import math
import re
from pathlib import Path

import chromadb
from pypdf import PdfReader

DOC_IDS = ["GL-GUIDELINES", "PROPERTY-APPETITE", "CYBER-EXCLUSIONS", "WORKCOMP-CLASSIFICATION"]


def load_pdf_doc(path: Path) -> dict:
    """Extract title + body text from one of our underwriting-manual PDFs."""
    reader = PdfReader(path)
    raw_text = "\n".join(page.extract_text() for page in reader.pages)
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]

    title = lines[1]
    body = " ".join(lines[3:])
    return {"title": title, "text": body}


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    return dot / (norm_a * norm_b)


def semantic_chunk(client, text: str, percentile: float = 40) -> list[str]:
    """Group sentences into chunks based on embedding similarity, splitting where
    similarity drops below the given percentile of this document's own
    sentence-to-sentence similarity distribution - not a fixed size or a fixed
    absolute similarity value.
    """
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]
    if len(sentences) <= 1:
        return sentences

    embeddings = client.embed(sentences)
    similarities = [
        cosine_similarity(embeddings[i - 1], embeddings[i])
        for i in range(1, len(sentences))
    ]

    threshold = sorted(similarities)[int(len(similarities) * percentile / 100)]

    chunks = []
    current_chunk = [sentences[0]]
    for i, similarity in enumerate(similarities, start=1):
        if similarity >= threshold:
            current_chunk.append(sentences[i])
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentences[i]]
    chunks.append(" ".join(current_chunk))
    return chunks


def build_underwriting_index(client, docs_dir: Path = Path("../session-2-rag-retrieval/docs")):
    """Load the 4 underwriting PDFs, semantically chunk them, and index them in a fresh
    in-memory Chroma collection.

    Returns `(collection, underwriting_docs)` - a fresh collection is built every call
    since Chroma's in-memory client doesn't persist across kernels/notebooks.
    """
    underwriting_docs = {}
    for doc_id in DOC_IDS:
        filename = doc_id.lower() + ".pdf"
        underwriting_docs[doc_id] = load_pdf_doc(docs_dir / filename)

    all_chunks = []
    for doc_id, doc in underwriting_docs.items():
        for i, chunk in enumerate(semantic_chunk(client, doc["text"])):
            all_chunks.append({"text": chunk, "source": doc_id, "chunk_index": i})

    chroma_client = chromadb.Client()
    collection = chroma_client.get_or_create_collection("underwriting_docs")

    chunk_texts = [c["text"] for c in all_chunks]
    chunk_embeddings = client.embed(chunk_texts)

    collection.add(
        ids=[f"{c['source']}-{c['chunk_index']}" for c in all_chunks],
        embeddings=chunk_embeddings,
        documents=chunk_texts,
        metadatas=[{"source": c["source"], "chunk_index": c["chunk_index"]} for c in all_chunks],
    )

    return collection, underwriting_docs
