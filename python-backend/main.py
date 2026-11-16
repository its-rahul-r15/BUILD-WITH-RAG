"""
main.py - The FastAPI server (the "front door" of the Python backend).

It receives requests from the frontend and calls the right functions:
  pdf.py       -> read the PDF
  rag.py       -> chunks, embeddings, Gemini
  database.py  -> ChromaDB
"""

import os
import uuid

from dotenv import load_dotenv

# Load the variables from the .env file BEFORE importing our own modules
load_dotenv()

import uvicorn
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import pdf
import rag
import database

MAX_FILE_SIZE_MB = 10
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

app = FastAPI(title="DocMind AI - Python Backend")

# CORS: the browser blocks a page from calling another address unless the server allows it.
# For this workshop we allow any origin. In a real product, list only your own website here.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def error_response(message, status_code):
    """Every error uses the same JSON shape in both backends."""
    return JSONResponse(status_code=status_code, content={"success": False, "error": message})


@app.exception_handler(RequestValidationError)
async def handle_bad_request(request: Request, error: RequestValidationError):
    return error_response("Invalid request. Please check the data you sent.", 400)


class ChatRequest(BaseModel):
    question: str


# ---------------------------------------------------------------
# GET /api/health
# ---------------------------------------------------------------
@app.get("/api/health")
def health():
    return {
        "success": True,
        "status": "ok",
        "backend": "python",
        "gemini_key_configured": bool(os.getenv("GEMINI_API_KEY")),
    }


# ---------------------------------------------------------------
# POST /api/documents/upload   (Ingestion pipeline)
# ---------------------------------------------------------------
@app.post("/api/documents/upload")
def upload_document(file: UploadFile = File(...)):
    # Validate the file name. os.path.basename removes any folder path a client may send.
    filename = os.path.basename(file.filename or "")
    if not filename.lower().endswith(".pdf"):
        return error_response("Only PDF files are allowed.", 400)

    # Read the file, but never more than the limit (+1 byte so we can detect "too big")
    pdf_bytes = file.file.read(MAX_FILE_SIZE_BYTES + 1)
    if len(pdf_bytes) > MAX_FILE_SIZE_BYTES:
        return error_response("File is too large. Maximum size is " + str(MAX_FILE_SIZE_MB) + " MB.", 400)
    if len(pdf_bytes) == 0:
        return error_response("The uploaded file is empty.", 400)

    try:
        # Step 1: Extract text from the uploaded PDF (page by page)
        # We need the original text before creating embeddings.
        pages, total_pages = pdf.extract_pages(pdf_bytes)

        # Step 2: Divide the text into smaller chunks
        # Smaller chunks help retrieve relevant information accurately.
        chunks = rag.split_into_chunks(pages)

        # Step 3: Generate embeddings
        # Embeddings convert text into numerical vectors that capture meaning.
        texts = [chunk["text"] for chunk in chunks]
        embeddings = rag.create_embeddings(texts, "RETRIEVAL_DOCUMENT")

        # Step 4: Store chunks + embeddings in ChromaDB
        # Now we can search them later by meaning.
        document_id = uuid.uuid4().hex  # our own id - we never trust names from the client
        database.store_chunks(document_id, filename, total_pages, chunks, embeddings)

    except ValueError as error:
        # Friendly problems: invalid PDF, empty PDF, password protected...
        return error_response(str(error), 400)
    except RuntimeError as error:
        # Problems with Gemini, ChromaDB or a missing API key
        return error_response(str(error), 500)
    except Exception as error:
        return error_response("Unexpected error: " + str(error), 500)

    return {
        "success": True,
        "document": {
            "id": document_id,
            "filename": filename,
            "pages": total_pages,
            "chunks": len(chunks),
            "status": "ready",
        },
    }


# ---------------------------------------------------------------
# GET /api/documents
# ---------------------------------------------------------------
@app.get("/api/documents")
def get_documents():
    try:
        documents = database.list_documents()
    except RuntimeError as error:
        return error_response(str(error), 500)
    return {"success": True, "documents": documents}


# ---------------------------------------------------------------
# POST /api/chat   (Question answering pipeline)
# ---------------------------------------------------------------
@app.post("/api/chat")
def chat(request: ChatRequest):
    question = request.question.strip()
    if question == "":
        return error_response("Please type a question.", 400)
    if len(question) > 2000:
        return error_response("The question is too long (maximum 2000 characters).", 400)

    try:
        result = rag.answer_question(question)
    except RuntimeError as error:
        return error_response(str(error), 500)
    except Exception as error:
        return error_response("Unexpected error: " + str(error), 500)

    return {"success": True, "answer": result["answer"], "sources": result["sources"]}


# ---------------------------------------------------------------
# DELETE /api/documents/{document_id}
# ---------------------------------------------------------------
@app.delete("/api/documents/{document_id}")
def delete_document(document_id: str):
    try:
        deleted_chunks = database.delete_document(document_id)
    except RuntimeError as error:
        return error_response(str(error), 500)

    if deleted_chunks == 0:
        return error_response("Document not found.", 404)

    return {"success": True, "deleted_chunks": deleted_chunks}


# Lets you start the server with:  python main.py
if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
