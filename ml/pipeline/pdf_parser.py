"""
ml/pipeline/pdf_parser.py

Extract text and headings from PDF (and plain-text) documents using PyMuPDF.

Output per page:
  {
    "page":    int,          # 1-based page number
    "heading": str | None,   # dominant heading on the page (largest font span)
    "text":    str,          # all body text, whitespace-normalised
    "blocks":  list[dict],   # raw block data for downstream use
  }
"""

import logging
import os
import re
from typing import List, Optional

logger = logging.getLogger(__name__)

try:
    import fitz  # PyMuPDF
    _FITZ_AVAILABLE = True
except ImportError:  # pragma: no cover
    _FITZ_AVAILABLE = False
    logger.warning("PyMuPDF (fitz) not installed; PDF parsing will be limited.")


class PDFParser:
    """
    Parse a PDF (or .txt/.md) file into a list of page dicts.

    Usage:
        parser = PDFParser()
        pages = parser.parse("/tmp/my_doc.pdf")
    """

    # Font-size threshold above which a span is treated as a heading candidate
    HEADING_FONT_THRESHOLD_PT = 13.0
    # Maximum heading string length (prevents long paragraphs being mislabelled)
    MAX_HEADING_LEN = 120
    # Minimum body text length per page to be considered non-empty
    MIN_PAGE_TEXT_LEN = 20

    def parse(self, file_path: str) -> List[dict]:
        """
        Parse *file_path* and return a list of page dicts.

        Supports:
          - .pdf  — full text + heading extraction via PyMuPDF
          - .txt / .md — split by double-newline into "virtual pages"
          - .docx — falls back to python-docx if available

        Args:
            file_path: Absolute path to the file.

        Returns:
            List of page dicts (see module docstring).
        """
        ext = os.path.splitext(file_path)[1].lower()

        if ext == ".pdf":
            return self._parse_pdf(file_path)
        elif ext in (".txt", ".md"):
            return self._parse_text(file_path)
        elif ext == ".docx":
            return self._parse_docx(file_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")

    # ------------------------------------------------------------------
    # PDF
    # ------------------------------------------------------------------

    def _parse_pdf(self, file_path: str) -> List[dict]:
        if not _FITZ_AVAILABLE:
            raise RuntimeError("PyMuPDF is required for PDF parsing. Install it with: pip install pymupdf")

        pages = []
        doc = fitz.open(file_path)

        for page_num in range(len(doc)):
            page = doc[page_num]
            blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]

            heading = self._extract_heading(blocks)
            body_text = self._extract_body_text(blocks, heading)

            if len(body_text.strip()) < self.MIN_PAGE_TEXT_LEN:
                # Skip image-only or near-empty pages
                continue

            pages.append({
                "page":    page_num + 1,
                "heading": heading,
                "text":    body_text,
                "blocks":  blocks,
            })

        doc.close()
        logger.debug("PDF parsed: %d content pages from %s", len(pages), file_path)
        return pages

    def _extract_heading(self, blocks: list) -> Optional[str]:
        """
        Find the largest-font span on the page that looks like a heading.
        Returns None if no suitable span is found.
        """
        best_size = 0.0
        best_text = None

        for block in blocks:
            if block.get("type") != 0:  # 0 = text block
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    size = span.get("size", 0)
                    text = span.get("text", "").strip()
                    if (
                        size > self.HEADING_FONT_THRESHOLD_PT
                        and size > best_size
                        and 3 < len(text) <= self.MAX_HEADING_LEN
                        and not text.endswith(".")   # headings rarely end in period
                    ):
                        best_size = size
                        best_text = text

        return best_text

    def _extract_body_text(self, blocks: list, heading: Optional[str]) -> str:
        """
        Concatenate all text spans, stripping the heading line to avoid duplication.
        Normalise whitespace.
        """
        lines = []
        for block in blocks:
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                line_text = " ".join(
                    span.get("text", "") for span in line.get("spans", [])
                ).strip()
                if line_text and line_text != heading:
                    lines.append(line_text)

        raw = " ".join(lines)
        # Collapse excessive whitespace
        clean = re.sub(r"\s{2,}", " ", raw).strip()
        return clean

    # ------------------------------------------------------------------
    # Plain text / Markdown
    # ------------------------------------------------------------------

    def _parse_text(self, file_path: str) -> List[dict]:
        with open(file_path, "r", encoding="utf-8", errors="replace") as fh:
            content = fh.read()

        # Split by blank lines into virtual "pages"
        sections = re.split(r"\n{2,}", content.strip())
        pages = []
        for idx, section in enumerate(sections):
            if not section.strip():
                continue
            lines = section.strip().splitlines()
            # First line of each section becomes the "heading" if short
            heading = lines[0].strip().lstrip("#").strip() if lines else None
            if heading and len(heading) > self.MAX_HEADING_LEN:
                heading = None
            body = " ".join(l.strip() for l in lines[1:] if l.strip()) or section.strip()
            pages.append({
                "page":    idx + 1,
                "heading": heading,
                "text":    re.sub(r"\s{2,}", " ", body),
                "blocks":  [],
            })
        return pages

    # ------------------------------------------------------------------
    # DOCX (python-docx fallback)
    # ------------------------------------------------------------------

    def _parse_docx(self, file_path: str) -> List[dict]:
        try:
            from docx import Document as DocxDocument
        except ImportError:
            raise RuntimeError("python-docx required for .docx parsing. pip install python-docx")

        docx = DocxDocument(file_path)
        pages, current_heading, current_lines = [], None, []

        def _flush(page_num):
            nonlocal current_heading, current_lines
            if current_lines:
                pages.append({
                    "page":    page_num,
                    "heading": current_heading,
                    "text":    " ".join(current_lines),
                    "blocks":  [],
                })
            current_heading, current_lines = None, []

        virtual_page = 1
        for para in docx.paragraphs:
            text = para.text.strip()
            if not text:
                continue
            if para.style.name.startswith("Heading"):
                _flush(virtual_page)
                virtual_page += 1
                current_heading = text
            else:
                current_lines.append(text)

        _flush(virtual_page)
        return pages