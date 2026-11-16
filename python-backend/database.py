"""
database.py - Everything related to ChromaDB (our vector database).

A vector database stores text together with its embedding (a list of numbers).
It can quickly find the stored texts whose numbers are closest to a question's numbers.
That is called "similarity search", and it is the heart of RAG.
"""

import os
import chromadb

# Folder where ChromaDB saves its files, so data survives a server restart.
CHROMA_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_data")

_collection = None  # we create the collection once and reuse it


def get_collection():
    """Opens ChromaDB (creates it the first time) and returns our collection."""
    global _collection

    if _collection is None:
        try:
            # PersistentClient = ChromaDB that stores everything in a local folder
            client = chromadb.PersistentClient(path=CHROMA_FOLDER)

            # A collection is like a table. "cosine" tells Chroma to compare
            # vectors by the angle between them, which works well for text.
            _collection = client.get_or_create_collection(
                name="documents",
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as error:
            raise RuntimeError("Could not open ChromaDB: " + str(error))

    return _collection


def store_chunks(document_id, filename, total_pages, chunks, embeddings):
    """
    Saves every chunk with its embedding.
    chunks     = [{"page": 1, "text": "..."}, ...]
    embeddings = one list of numbers per chunk (same order as chunks)
    """
    collection = get_collection()

    ids = []
    texts = []
    metadatas = []

    for number, chunk in enumerate(chunks):
        ids.append(document_id + "_" + str(number))  # every chunk needs a unique id
        texts.append(chunk["text"])
        # Metadata is extra information we get back together with the text.
        # This is how we know the filename and page number of each chunk.
        metadatas.append({
            "document_id": document_id,
            "filename": filename,
            "page": chunk["page"],
            "total_pages": total_pages,
        })

    collection.add(ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas)


def search_similar(query_embedding, how_many=4):
    """
    Finds the chunks closest to the question embedding.
    Returns a list like: [{"text": "...", "filename": "a.pdf", "page": 2}, ...]
    """
    collection = get_collection()

    total_chunks = collection.count()
    if total_chunks == 0:
        return []

    # We cannot ask for more results than the number of stored chunks
    how_many = min(how_many, total_chunks)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=how_many,
        include=["documents", "metadatas"],
    )

    # Chroma returns lists inside lists (one list per query). We sent one query, so we take [0].
    texts = results["documents"][0]
    metadatas = results["metadatas"][0]

    found = []
    for text, metadata in zip(texts, metadatas):
        found.append({
            "text": text,
            "filename": metadata["filename"],
            "page": metadata["page"],
        })
    return found


def list_documents():
    """
    Returns one entry per uploaded document.
    We do not keep a separate documents table - we simply group the chunks by document_id.
    """
    collection = get_collection()
    stored = collection.get(include=["metadatas"])

    documents = {}
    for metadata in stored["metadatas"]:
        document_id = metadata["document_id"]
        if document_id not in documents:
            documents[document_id] = {
                "id": document_id,
                "filename": metadata["filename"],
                "pages": metadata["total_pages"],
                "chunks": 0,
            }
        documents[document_id]["chunks"] += 1

    return list(documents.values())


def delete_document(document_id):
    """Deletes all chunks (vectors) of one document. Returns how many were deleted."""
    collection = get_collection()

    existing = collection.get(where={"document_id": document_id}, include=[])
    count = len(existing["ids"])

    if count > 0:
        collection.delete(where={"document_id": document_id})

    return count
