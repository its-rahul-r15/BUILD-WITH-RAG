"""
pdf.py - Reads a PDF and gives us the text of every page.

Why do we keep page numbers?
Later, when the AI answers, we want to tell the user
"this answer came from page 3 of syllabus.pdf".
"""

import pymupdf  # PyMuPDF library (it is imported with the name "pymupdf")


def extract_pages(pdf_bytes):
    """
    Takes the raw bytes of a PDF file.
    Returns a list like: [{"page": 1, "text": "..."}, {"page": 2, "text": "..."}]
    Pages that contain no text are skipped.
    Raises ValueError with a friendly message if the PDF is invalid or empty.
    """

    # A real PDF always starts with these 5 characters.
    # This is a quick check before we even try to open the file.
    if not pdf_bytes.startswith(b"%PDF"):
        raise ValueError("This file is not a valid PDF.")

    # Open the PDF directly from memory. We never save the uploaded file on disk,
    # so we never have to trust the file name or path sent by the user.
    try:
        document = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    except Exception:
        raise ValueError("This PDF is corrupted and could not be opened.")

    if document.needs_pass:
        document.close()
        raise ValueError("This PDF is password protected. Please upload an unlocked PDF.")

    pages = []
    for index in range(len(document)):
        page_text = document[index].get_text().strip()

        # Skip empty pages (for example, pages that only contain images)
        if page_text == "":
            continue

        # PyMuPDF counts pages from 0, humans count from 1
        pages.append({"page": index + 1, "text": page_text})

    total_pages = len(document)
    document.close()

    if len(pages) == 0:
        raise ValueError(
            "No text found in this PDF. It may be a scanned document (images only)."
        )

    return pages, total_pages
