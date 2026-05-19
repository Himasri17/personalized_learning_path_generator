"""
ml/pipeline/nlp_processor.py

NLP post-processing of parsed pages using spaCy.

Responsibilities
----------------
1. Sentence-aware chunking — splits body text into overlapping windows
   so each chunk fits within the embedding model's token limit (≤256 tokens).
2. Topic extraction — noun-chunk and entity-based topic labels.
3. Named-entity extraction — person, org, tech, concept labels.
4. Deduplication — drops near-identical chunks via exact-text hashing.

Output per chunk:
  {
    "chunk_id": "<doc_page>_<idx>",
    "page":     int,
    "heading":  str | None,
    "text":     str,              # the chunk text
    "topics":   list[str],        # top noun-chunk lemmas
    "entities": list[str],        # NER labels formatted as "text (LABEL)"
  }
"""

import hashlib
import logging
import re
from typing import List, Optional

logger = logging.getLogger(__name__)

try:
    import spacy
    _SPACY_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SPACY_AVAILABLE = False
    logger.warning("spaCy not installed; NLP features will be degraded.")


class NLPProcessor:
    """
    Process a list of page dicts (from PDFParser) into enriched text chunks.

    Args:
        model_name:    spaCy model to load (default: en_core_web_sm).
                       Use en_core_web_md or en_core_web_lg for better NER.
        chunk_size:    Target number of sentences per chunk.
        chunk_overlap: Number of sentences to overlap between adjacent chunks.
        max_topics:    Maximum number of topic labels to extract per chunk.

    Usage:
        nlp = NLPProcessor()
        chunks = nlp.process(pages)
    """

    def __init__(
        self,
        model_name: str = "en_core_web_sm",
        chunk_size: int = 6,
        chunk_overlap: int = 1,
        max_topics: int = 5,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_topics = max_topics
        self._seen_hashes: set = set()

        if _SPACY_AVAILABLE:
            try:
                self._nlp = spacy.load(model_name, disable=["parser"])
                # Enable sentenciser via senter (lighter than full parser)
                if "senter" not in self._nlp.pipe_names:
                    self._nlp.add_pipe("senter")
            except OSError:
                logger.warning(
                    "spaCy model '%s' not found. Run: python -m spacy download %s",
                    model_name, model_name,
                )
                self._nlp = None
        else:
            self._nlp = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, pages: List[dict]) -> List[dict]:
        """
        Convert page dicts into enriched chunk dicts.

        Args:
            pages: Output of PDFParser.parse().

        Returns:
            List of chunk dicts (see module docstring).
        """
        self._seen_hashes.clear()
        all_chunks = []

        for page in pages:
            page_num = page["page"]
            heading  = page.get("heading")
            text     = page.get("text", "").strip()

            if not text:
                continue

            sentences = self._sentencise(text)
            windows   = self._sliding_windows(sentences)

            for idx, window_sents in enumerate(windows):
                chunk_text = " ".join(window_sents).strip()
                chunk_text = re.sub(r"\s{2,}", " ", chunk_text)

                if not chunk_text or self._is_duplicate(chunk_text):
                    continue

                topics, entities = self._extract_metadata(chunk_text)

                all_chunks.append({
                    "chunk_id": f"p{page_num}_{idx}",
                    "page":     page_num,
                    "heading":  heading,
                    "text":     chunk_text,
                    "topics":   topics,
                    "entities": entities,
                })

        logger.debug("NLPProcessor produced %d chunks from %d pages.", len(all_chunks), len(pages))
        return all_chunks

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _sentencise(self, text: str) -> List[str]:
        """Split text into sentences using spaCy senter or regex fallback."""
        if self._nlp:
            doc = self._nlp(text[:100_000])  # hard cap for very large pages
            return [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        # Regex fallback — split on ". ", "! ", "? "
        raw = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in raw if s.strip()]

    def _sliding_windows(self, sentences: List[str]) -> List[List[str]]:
        """
        Produce overlapping windows of sentences.

        Example (chunk_size=3, overlap=1):
          [s0,s1,s2], [s2,s3,s4], [s4,s5,s6], ...
        """
        if not sentences:
            return []

        step = max(1, self.chunk_size - self.chunk_overlap)
        windows = []
        i = 0
        while i < len(sentences):
            window = sentences[i: i + self.chunk_size]
            windows.append(window)
            if i + self.chunk_size >= len(sentences):
                break
            i += step
        return windows

    def _extract_metadata(self, text: str):
        """
        Return (topics, entities) for a chunk text.
        Falls back to empty lists if spaCy is unavailable.
        """
        topics: List[str] = []
        entities: List[str] = []

        if not self._nlp:
            return topics, entities

        doc = self._nlp(text[:5_000])

        # Topics: top noun-chunk lemmas (lower-cased, de-duplicated)
        seen_topics: set = set()
        for nc in doc.noun_chunks:
            lemma = nc.root.lemma_.lower().strip()
            if (
                lemma
                and lemma not in seen_topics
                and len(lemma) > 2
                and not nc.root.is_stop
            ):
                seen_topics.add(lemma)
                topics.append(lemma)
                if len(topics) >= self.max_topics:
                    break

        # Entities: "surface_text (LABEL)" — deduplicated
        seen_ents: set = set()
        for ent in doc.ents:
            key = ent.text.strip().lower()
            if key and key not in seen_ents:
                seen_ents.add(key)
                entities.append(f"{ent.text.strip()} ({ent.label_})")

        return topics, entities

    def _is_duplicate(self, text: str) -> bool:
        """Deduplicate chunks by SHA-256 hash of normalised text."""
        normalised = re.sub(r"\s+", " ", text.lower().strip())
        h = hashlib.sha256(normalised.encode()).hexdigest()
        if h in self._seen_hashes:
            return True
        self._seen_hashes.add(h)
        return False