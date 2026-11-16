// server.js - The Express server (the "front door" of the JavaScript backend).
//
// It receives requests from the frontend and calls the right functions:
//   pdf.js       -> read the PDF
//   rag.js       -> chunks, embeddings, Gemini
//   database.js  -> ChromaDB

import "dotenv/config"; // loads the .env file into process.env (must come first)

import crypto from "node:crypto";
import path from "node:path";
import express from "express";
import cors from "cors";
import multer from "multer";

import { extractPages } from "./pdf.js";
import * as rag from "./rag.js";
import * as database from "./database.js";

const PORT = Number(process.env.PORT || 3000);
const MAX_FILE_SIZE_MB = 10;

const app = express();

// CORS: the browser blocks a page from calling another address unless the server allows it.
// For this workshop we allow any origin. In a real product, list only your own website here.
app.use(cors());
app.use(express.json({ limit: "100kb" }));

// multer reads uploaded files. memoryStorage keeps the file in RAM (no file is saved on disk),
// so we never have to trust the file name or path sent by the user.
const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: MAX_FILE_SIZE_MB * 1024 * 1024 },
});

// Every error uses the same JSON shape in both backends.
function sendError(res, message, statusCode) {
  res.status(statusCode).json({ success: false, error: message });
}

// ---------------------------------------------------------------
// GET /api/health
// ---------------------------------------------------------------
app.get("/api/health", function (req, res) {
  res.json({
    success: true,
    status: "ok",
    backend: "javascript",
    gemini_key_configured: Boolean(process.env.GEMINI_API_KEY),
  });
});

// ---------------------------------------------------------------
// POST /api/documents/upload   (Ingestion pipeline)
// ---------------------------------------------------------------
app.post("/api/documents/upload", upload.single("file"), async function (req, res) {
  if (!req.file) {
    return sendError(res, "No file received. Please choose a PDF.", 400);
  }

  // Validate the file name. path.basename removes any folder path a client may send.
  const filename = path.basename(req.file.originalname || "");
  if (!filename.toLowerCase().endsWith(".pdf")) {
    return sendError(res, "Only PDF files are allowed.", 400);
  }
  if (req.file.size === 0) {
    return sendError(res, "The uploaded file is empty.", 400);
  }

  try {
    // Step 1: Extract text from the uploaded PDF (page by page)
    // We need the original text before creating embeddings.
    const { pages, totalPages } = await extractPages(req.file.buffer);

    // Step 2: Divide the text into smaller chunks
    // Smaller chunks help retrieve relevant information accurately.
    const chunks = rag.splitIntoChunks(pages);

    // Step 3: Generate embeddings
    // Embeddings convert text into numerical vectors that capture meaning.
    const texts = chunks.map(function (chunk) { return chunk.text; });
    const embeddings = await rag.createEmbeddings(texts, "RETRIEVAL_DOCUMENT");

    // Step 4: Store chunks + embeddings in ChromaDB
    // Now we can search them later by meaning.
    const documentId = crypto.randomUUID().replaceAll("-", ""); // our own id - never trust client names
    await database.storeChunks(documentId, filename, totalPages, chunks, embeddings);

    res.json({
      success: true,
      document: {
        id: documentId,
        filename: filename,
        pages: totalPages,
        chunks: chunks.length,
        status: "ready",
      },
    });
  } catch (error) {
    // Messages from our own modules are already friendly (invalid PDF, Gemini error, ChromaDB down...)
    sendError(res, error.message, 400);
  }
});

// ---------------------------------------------------------------
// GET /api/documents
// ---------------------------------------------------------------
app.get("/api/documents", async function (req, res) {
  try {
    const documents = await database.listDocuments();
    res.json({ success: true, documents: documents });
  } catch (error) {
    sendError(res, error.message, 500);
  }
});

// ---------------------------------------------------------------
// POST /api/chat   (Question answering pipeline)
// ---------------------------------------------------------------
app.post("/api/chat", async function (req, res) {
  const question = String((req.body && req.body.question) || "").trim();

  if (question === "") {
    return sendError(res, "Please type a question.", 400);
  }
  if (question.length > 2000) {
    return sendError(res, "The question is too long (maximum 2000 characters).", 400);
  }

  try {
    const result = await rag.answerQuestion(question);
    res.json({ success: true, answer: result.answer, sources: result.sources });
  } catch (error) {
    sendError(res, error.message, 500);
  }
});

// ---------------------------------------------------------------
// DELETE /api/documents/:id
// ---------------------------------------------------------------
app.delete("/api/documents/:id", async function (req, res) {
  try {
    const deletedChunks = await database.deleteDocument(req.params.id);

    if (deletedChunks === 0) {
      return sendError(res, "Document not found.", 404);
    }
    res.json({ success: true, deleted_chunks: deletedChunks });
  } catch (error) {
    sendError(res, error.message, 500);
  }
});

// ---------------------------------------------------------------
// Error handler for problems that happen before our routes run
// (for example: file bigger than the limit, or broken JSON)
// ---------------------------------------------------------------
app.use(function (error, req, res, next) {
  if (error instanceof multer.MulterError && error.code === "LIMIT_FILE_SIZE") {
    return sendError(res, "File is too large. Maximum size is " + MAX_FILE_SIZE_MB + " MB.", 400);
  }
  if (error instanceof multer.MulterError) {
    return sendError(res, "Upload error: " + error.message, 400);
  }
  sendError(res, "Invalid request.", 400);
});

app.listen(PORT, function () {
  console.log("DocMind JavaScript backend running on http://localhost:" + PORT);
  if (!process.env.GEMINI_API_KEY) {
    console.log("WARNING: GEMINI_API_KEY is missing. Add it to your .env file.");
  }
});
