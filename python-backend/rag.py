"""
rag.py - The RAG logic.

RAG = Retrieval-Augmented Generation
  1. Retrieval  : find the parts of the documents that are related to the question
  2. Augmented  : add those parts to the prompt
  3. Generation : let Gemini write the answer using only that information
"""

import os
from google import genai
from google.genai import types

import database

# Model names can be changed from the .env file without touching the code.
EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-3.6-flash")

CHUNK_SIZE = 800      # characters per chunk
CHUNK_OVERLAP = 150   # characters shared between two neighbouring chunks
TOP_K = 4             # how many chunks we send to Gemini

NOT_FOUND_MESSAGE = "I could not find enough information in the uploaded documents to answer this question."

_client = None


def get_gemini_client():
    """Creates the Gemini client once. The API key comes from the environment (.env file)."""
    global _client

    if _client is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is missing. Add it to your .env file and restart the server.")
        _client = genai.Client(api_key=api_key)

    return _client


# ---------------------------------------------------------------
# Step 2 of ingestion: split text into chunks
# ---------------------------------------------------------------
def split_into_chunks(pages):
    """
    Splits every page into smaller pieces (chunks).

    Why? A whole page contains many topics. A small chunk contains one idea,
    so the search can find exactly the right piece.
    Why overlap? So a sentence cut at the end of one chunk still appears
    fully at the start of the next chunk.
    """
    chunks = []

    for page in pages:
        text = page["text"]
        start = 0

        while start < len(text):
            end = start + CHUNK_SIZE
            piece = text[start:end].strip()

            if piece != "":
                chunks.append({"page": page["page"], "text": piece})

            if end >= len(text):
                break  # we reached the end of this page

            # Move forward, but step back a little to create the overlap
            start = end - CHUNK_OVERLAP

    return chunks


# ---------------------------------------------------------------
# Step 3 of ingestion: embeddings
# ---------------------------------------------------------------
def create_embeddings(texts, task_type):
    """
    Converts a list of texts into a list of embeddings.

    An embedding is a long list of numbers that represents the MEANING of a text.
    Texts with similar meaning get similar numbers.

    task_type tells Gemini what the embedding is for:
      "RETRIEVAL_DOCUMENT" -> for the document chunks we store
      "RETRIEVAL_QUERY"    -> for the user's question
    """
    client = get_gemini_client()
    all_embeddings = []

    # We send the texts in small batches because the API accepts limited input per request
    batch_size = 50
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]

        try:
            response = client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=batch,
                config=types.EmbedContentConfig(
                    task_type=task_type,
                    output_dimensionality=768,  # smaller vectors = less storage
                ),
            )
        except Exception as error:
            raise RuntimeError("Gemini embedding error: " + str(error))

        for item in response.embeddings:
            all_embeddings.append(list(item.values))

    return all_embeddings


# ---------------------------------------------------------------
# Question answering pipeline
# ---------------------------------------------------------------
def build_prompt(question, found_chunks):
    """Puts the retrieved chunks and the question together into one prompt for Gemini."""

    context_parts = []
    for chunk in found_chunks:
        label = "[Source: " + chunk["filename"] + ", Page " + str(chunk["page"]) + "]"
        context_parts.append(label + "\n" + chunk["text"])
    context = "\n\n".join(context_parts)

    # The instructions below stop the AI from making things up (hallucinating)
    prompt = (
        "You are DocMind AI, an assistant that answers questions about uploaded documents.\n"
        "Answer the question using ONLY the context below.\n"
        "If the context does not contain the answer, reply exactly with this sentence:\n"
        + NOT_FOUND_MESSAGE + "\n\n"
        "CONTEXT:\n" + context + "\n\n"
        "QUESTION: " + question + "\n\n"
        "ANSWER:"
    )
    return prompt


def answer_question(question):
    """
    Full question answering pipeline.
    Returns {"answer": "...", "sources": [{"filename": ..., "page": ..., "snippet": ...}]}
    """

    # Step 1: Convert the question into an embedding
    # It must use the same model as the document chunks so both live in the same "number space".
    question_embedding = create_embeddings([question], "RETRIEVAL_QUERY")[0]

    # Step 2: Search ChromaDB for the most similar chunks
    # This is the "Retrieval" in RAG. We do NOT send the whole PDF to Gemini.
    found_chunks = database.search_similar(question_embedding, TOP_K)

    if len(found_chunks) == 0:
        return {"answer": "No documents are uploaded yet. Please upload a PDF first.", "sources": []}

    # Step 3: Build the prompt = instructions + retrieved chunks + question
    # This is the "Augmented" in RAG.
    prompt = build_prompt(question, found_chunks)

    # Step 4: Ask Gemini to write the answer. This is the "Generation" in RAG.
    client = get_gemini_client()
    try:
        response = client.models.generate_content(model=CHAT_MODEL, contents=prompt)
    except Exception as error:
        raise RuntimeError("Gemini error: " + str(error))

    answer = (response.text or "").strip()
    if answer == "":
        answer = NOT_FOUND_MESSAGE

    # Step 5: Prepare the sources (filename + page). Remove duplicates.
    sources = []
    already_added = set()
    for chunk in found_chunks:
        key = (chunk["filename"], chunk["page"])
        if key in already_added:
            continue
        already_added.add(key)
        sources.append({
            "filename": chunk["filename"],
            "page": chunk["page"],
            "snippet": chunk["text"][:150],
        })

    # If Gemini says the answer is not in the documents, showing sources would be misleading
    if NOT_FOUND_MESSAGE in answer:
        sources = []

    return {"answer": answer, "sources": sources}
