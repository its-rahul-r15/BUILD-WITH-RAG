// rag.js - The RAG logic.
//
// RAG = Retrieval-Augmented Generation
//   1. Retrieval  : find the parts of the documents that are related to the question
//   2. Augmented  : add those parts to the prompt
//   3. Generation : let Gemini write the answer using only that information

import { GoogleGenAI } from "@google/genai";
import * as database from "./database.js";

// Model names can be changed from the .env file without touching the code.
const EMBEDDING_MODEL = process.env.GEMINI_EMBEDDING_MODEL || "gemini-embedding-001";
const CHAT_MODEL = process.env.GEMINI_CHAT_MODEL || "gemini-3.6-flash";

const CHUNK_SIZE = 800;     // characters per chunk
const CHUNK_OVERLAP = 150;  // characters shared between two neighbouring chunks
const TOP_K = 4;            // how many chunks we send to Gemini

export const NOT_FOUND_MESSAGE =
  "I could not find enough information in the uploaded documents to answer this question.";

let client = null;

// Creates the Gemini client once. The API key comes from the environment (.env file).
function getGeminiClient() {
  if (client === null) {
    const apiKey = process.env.GEMINI_API_KEY;
    if (!apiKey) {
      throw new Error("GEMINI_API_KEY is missing. Add it to your .env file and restart the server.");
    }
    client = new GoogleGenAI({ apiKey: apiKey });
  }
  return client;
}

// ---------------------------------------------------------------
// Step 2 of ingestion: split text into chunks
// ---------------------------------------------------------------
// Splits every page into smaller pieces (chunks).
//
// Why? A whole page contains many topics. A small chunk contains one idea,
// so the search can find exactly the right piece.
// Why overlap? So a sentence cut at the end of one chunk still appears
// fully at the start of the next chunk.
export function splitIntoChunks(pages) {
  const chunks = [];

  for (const page of pages) {
    const text = page.text;
    let start = 0;

    while (start < text.length) {
      const end = start + CHUNK_SIZE;
      const piece = text.slice(start, end).trim();

      if (piece !== "") {
        chunks.push({ page: page.page, text: piece });
      }

      if (end >= text.length) {
        break; // we reached the end of this page
      }

      // Move forward, but step back a little to create the overlap
      start = end - CHUNK_OVERLAP;
    }
  }

  return chunks;
}

// ---------------------------------------------------------------
// Step 3 of ingestion: embeddings
// ---------------------------------------------------------------
// Converts a list of texts into a list of embeddings.
//
// An embedding is a long list of numbers that represents the MEANING of a text.
// Texts with similar meaning get similar numbers.
//
// taskType tells Gemini what the embedding is for:
//   "RETRIEVAL_DOCUMENT" -> for the document chunks we store
//   "RETRIEVAL_QUERY"    -> for the user's question
export async function createEmbeddings(texts, taskType) {
  const ai = getGeminiClient();
  const allEmbeddings = [];

  // We send the texts in batches of 100 (the maximum supported in one batch call)
  const batchSize = 100;
  for (let start = 0; start < texts.length; start += batchSize) {
    if (start > 0) {
      // Small delay between batches to avoid tripping requests-per-minute rate limits
      await new Promise((resolve) => setTimeout(resolve, 600));
    }

    const batch = texts.slice(start, start + batchSize);

    let response;
    const maxRetries = 5;
    for (let attempt = 0; attempt < maxRetries; attempt++) {
      try {
        response = await ai.models.embedContent({
          model: EMBEDDING_MODEL,
          contents: batch,
          config: {
            taskType: taskType,
            outputDimensionality: 768, // smaller vectors = less storage
          },
        });
        break; // success
      } catch (error) {
        const errMsg = String(error.message || "");
        const isRateLimit = errMsg.includes("429") || errMsg.includes("quota") || errMsg.includes("RESOURCE_EXHAUSTED");

        if (isRateLimit && attempt < maxRetries - 1) {
          // Parse retry delay from error if present (e.g. "retry in 44.1s"), otherwise default to 45s
          let waitSeconds = 45;
          const match = errMsg.match(/retry in ([0-9.]+)s/i);
          if (match) {
            waitSeconds = Math.ceil(parseFloat(match[1])) + 2;
          }
          console.warn(`[Gemini Rate Limit] Waiting ${waitSeconds}s before retrying embedding batch (attempt ${attempt + 1}/${maxRetries})...`);
          await new Promise((resolve) => setTimeout(resolve, waitSeconds * 1000));
          continue;
        }

        throw new Error("Gemini embedding error: " + error.message);
      }
    }

    for (const item of response.embeddings) {
      allEmbeddings.push(item.values);
    }
  }

  return allEmbeddings;
}

// ---------------------------------------------------------------
// Question answering pipeline
// ---------------------------------------------------------------

// Puts the retrieved chunks and the question together into one prompt for Gemini.
function buildPrompt(question, foundChunks) {
  const contextParts = foundChunks.map(function (chunk) {
    const label = "[Source: " + chunk.filename + ", Page " + chunk.page + "]";
    return label + "\n" + chunk.text;
  });
  const context = contextParts.join("\n\n");

  // The instructions below stop the AI from making things up (hallucinating)
  return (
    "You are DocMind AI, an assistant that answers questions about uploaded documents.\n" +
    "Answer the question using ONLY the context below.\n" +
    "If the context does not contain the answer, reply exactly with this sentence:\n" +
    NOT_FOUND_MESSAGE + "\n\n" +
    "CONTEXT:\n" + context + "\n\n" +
    "QUESTION: " + question + "\n\n" +
    "ANSWER:"
  );
}

// Full question answering pipeline.
// Returns { answer: "...", sources: [{ filename, page, snippet }] }
export async function answerQuestion(question) {
  // Step 1: Convert the question into an embedding
  // It must use the same model as the document chunks so both live in the same "number space".
  const questionEmbeddings = await createEmbeddings([question], "RETRIEVAL_QUERY");
  const questionEmbedding = questionEmbeddings[0];

  // Step 2: Search ChromaDB for the most similar chunks
  // This is the "Retrieval" in RAG. We do NOT send the whole PDF to Gemini.
  const foundChunks = await database.searchSimilar(questionEmbedding, TOP_K);

  if (foundChunks.length === 0) {
    return { answer: "No documents are uploaded yet. Please upload a PDF first.", sources: [] };
  }

  // Step 3: Build the prompt = instructions + retrieved chunks + question
  // This is the "Augmented" in RAG.
  const prompt = buildPrompt(question, foundChunks);

  // Step 4: Ask Gemini to write the answer. This is the "Generation" in RAG.
  const ai = getGeminiClient();
  let response;
  try {
    response = await ai.models.generateContent({ model: CHAT_MODEL, contents: prompt });
  } catch (error) {
    throw new Error("Gemini error: " + error.message);
  }

  let answer = (response.text || "").trim();
  if (answer === "") {
    answer = NOT_FOUND_MESSAGE;
  }

  // Step 5: Prepare the sources (filename + page). Remove duplicates.
  let sources = [];
  const alreadyAdded = new Set();
  for (const chunk of foundChunks) {
    const key = chunk.filename + "|" + chunk.page;
    if (alreadyAdded.has(key)) continue;
    alreadyAdded.add(key);
    sources.push({
      filename: chunk.filename,
      page: chunk.page,
      snippet: chunk.text.slice(0, 150),
    });
  }

  // If Gemini says the answer is not in the documents, showing sources would be misleading
  if (answer.includes(NOT_FOUND_MESSAGE)) {
    sources = [];
  }

  return { answer: answer, sources: sources };
}
