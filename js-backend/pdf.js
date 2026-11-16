// pdf.js - Reads a PDF and gives us the text of every page.
//
// Why do we keep page numbers?
// Later, when the AI answers, we want to tell the user
// "this answer came from page 3 of syllabus.pdf".

import { PDFParse } from "pdf-parse";

// Takes the raw bytes (Buffer) of a PDF file.
// Returns { pages: [{ page: 1, text: "..." }, ...], totalPages: 10 }
// Throws an Error with a friendly message if the PDF is invalid or empty.
export async function extractPages(pdfBuffer) {
  // A real PDF always starts with "%PDF". Quick check before we try to open it.
  const startsLikePdf = pdfBuffer.subarray(0, 4).toString("latin1") === "%PDF";
  if (!startsLikePdf) {
    throw new Error("This file is not a valid PDF.");
  }

  // We read the PDF directly from memory. We never save the uploaded file on disk,
  // so we never have to trust the file name or path sent by the user.
  const parser = new PDFParse({ data: new Uint8Array(pdfBuffer) });

  let result;
  try {
    result = await parser.getText();
  } catch (error) {
    const message = String(error && error.message ? error.message : error);
    if (message.toLowerCase().includes("password")) {
      throw new Error("This PDF is password protected. Please upload an unlocked PDF.");
    }
    throw new Error("This PDF is corrupted and could not be opened.");
  } finally {
    await parser.destroy();
  }

  const pages = [];
  for (const item of result.pages) {
    const pageText = (item.text || "").trim();

    // Skip empty pages (for example, pages that only contain images)
    if (pageText === "") continue;

    pages.push({ page: item.num, text: pageText });
  }

  if (pages.length === 0) {
    throw new Error("No text found in this PDF. It may be a scanned document (images only).");
  }

  return { pages: pages, totalPages: result.total };
}
