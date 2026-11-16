// database.js - Everything related to ChromaDB (our vector database).
//
// A vector database stores text together with its embedding (a list of numbers).
// It can quickly find the stored texts whose numbers are closest to a question's numbers.
// That is called "similarity search", and it is the heart of RAG.
//
// IMPORTANT: the JavaScript ChromaDB package is only a CLIENT.
// It talks to a ChromaDB SERVER that you start separately:
//     chroma run --path ./chroma_data --port 8001
// The server saves the data in the "chroma_data" folder, so it is persistent.

import { ChromaClient } from "chromadb";

const CHROMA_HOST = process.env.CHROMA_HOST || "localhost";
const CHROMA_PORT = Number(process.env.CHROMA_PORT || 8001);

let collection = null; // we create the collection once and reuse it

// Connects to ChromaDB (first time only) and returns our collection.
async function getCollection() {
  if (collection !== null) {
    return collection;
  }

  try {
    const client = new ChromaClient({ host: CHROMA_HOST, port: CHROMA_PORT });

    // A collection is like a table. "cosine" tells Chroma to compare vectors by
    // the angle between them, which works well for text.
    // embeddingFunction: null -> we create the embeddings ourselves with Gemini.
    collection = await client.getOrCreateCollection({
      name: "documents",
      embeddingFunction: null,
      configuration: { hnsw: { space: "cosine" } },
    });
    return collection;
  } catch (error) {
    throw new Error(
      "Could not connect to ChromaDB at " + CHROMA_HOST + ":" + CHROMA_PORT +
      ". Start it with: chroma run --path ./chroma_data --port " + CHROMA_PORT
    );
  }
}

// Saves every chunk with its embedding.
// chunks     = [{ page: 1, text: "..." }, ...]
// embeddings = one array of numbers per chunk (same order as chunks)
export async function storeChunks(documentId, filename, totalPages, chunks, embeddings) {
  const col = await getCollection();

  const ids = [];
  const texts = [];
  const metadatas = [];

  chunks.forEach(function (chunk, number) {
    ids.push(documentId + "_" + number); // every chunk needs a unique id
    texts.push(chunk.text);
    // Metadata is extra information we get back together with the text.
    // This is how we know the filename and page number of each chunk.
    metadatas.push({
      document_id: documentId,
      filename: filename,
      page: chunk.page,
      total_pages: totalPages,
    });
  });

  await col.add({ ids: ids, documents: texts, embeddings: embeddings, metadatas: metadatas });
}

// Finds the chunks closest to the question embedding.
// Returns [{ text, filename, page }, ...]
export async function searchSimilar(queryEmbedding, howMany = 4) {
  const col = await getCollection();

  const totalChunks = await col.count();
  if (totalChunks === 0) {
    return [];
  }

  // We cannot ask for more results than the number of stored chunks
  const limit = Math.min(howMany, totalChunks);

  const results = await col.query({
    queryEmbeddings: [queryEmbedding],
    nResults: limit,
    include: ["documents", "metadatas"],
  });

  // Chroma returns arrays inside arrays (one per query). We sent one query, so we take [0].
  const texts = results.documents[0];
  const metadatas = results.metadatas[0];

  const found = [];
  for (let i = 0; i < texts.length; i++) {
    found.push({
      text: texts[i],
      filename: metadatas[i].filename,
      page: metadatas[i].page,
    });
  }
  return found;
}

// Returns one entry per uploaded document.
// We do not keep a separate documents table - we simply group the chunks by document_id.
export async function listDocuments() {
  const col = await getCollection();
  const stored = await col.get({ include: ["metadatas"] });

  const documents = {};
  for (const metadata of stored.metadatas) {
    const id = metadata.document_id;
    if (!documents[id]) {
      documents[id] = {
        id: id,
        filename: metadata.filename,
        pages: metadata.total_pages,
        chunks: 0,
      };
    }
    documents[id].chunks += 1;
  }

  return Object.values(documents);
}

// Deletes all chunks (vectors) of one document. Returns how many were deleted.
export async function deleteDocument(documentId) {
  const col = await getCollection();

  const existing = await col.get({ where: { document_id: documentId }, include: [] });
  const count = existing.ids.length;

  if (count > 0) {
    await col.delete({ where: { document_id: documentId } });
  }

  return count;
}
