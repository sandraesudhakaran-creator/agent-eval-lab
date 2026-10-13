"""Retrieval over a folder of Markdown documents.

Documents are split along their headings into small chunks, and chunks are ranked with
BM25, the classic keyword-relevance formula used by search engines. BM25 needs no model
download and is fully explainable, which makes it a solid, honest baseline; a dense
(embedding) retriever can be plugged in through the same ``search`` interface.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

_STOP = set("""a an and are as at be by can do does for from has have how i if in is it its
me my of on or our so than that the their them then there these this to was we what when
where which who why will with you your""".split())


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOP]


@dataclass
class Chunk:
    source: str      # document id, e.g. "warranty"
    section: str     # heading path, e.g. "Warranty > Battery"
    text: str

    @property
    def ref(self) -> str:
        return f"{self.source}#{self.section}"


def chunk_markdown(source: str, text: str, max_words: int = 140) -> list[Chunk]:
    """Split a Markdown document at its headings; long sections are split by paragraph."""
    chunks: list[Chunk] = []
    path: list[str] = []
    buf: list[str] = []

    def flush() -> None:
        body = "\n".join(buf).strip()
        buf.clear()
        if not body:
            return
        section = " > ".join(path) or source
        words, part = 0, []
        for para in re.split(r"\n\s*\n", body):
            n = len(para.split())
            if part and words + n > max_words:
                chunks.append(Chunk(source, section, "\n\n".join(part)))
                part, words = [], 0
            part.append(para)
            words += n
        if part:
            chunks.append(Chunk(source, section, "\n\n".join(part)))

    for line in text.splitlines():
        m = re.match(r"^(#{1,4})\s+(.*)", line)
        if m:
            flush()
            level = len(m.group(1))
            path[:] = path[: level - 1] + [m.group(2).strip()]
        else:
            buf.append(line)
    flush()
    return chunks


def load_markdown_dir(folder: str | Path, max_words: int = 140) -> list[Chunk]:
    chunks: list[Chunk] = []
    for p in sorted(Path(folder).glob("*.md")):
        chunks += chunk_markdown(p.stem, p.read_text(encoding="utf-8"), max_words)
    if not chunks:
        raise FileNotFoundError(f"No Markdown documents found in {folder}")
    return chunks


@dataclass
class Hit:
    chunk: Chunk
    score: float


class BM25Index:
    """Okapi BM25 over chunk text (section titles are included and count once)."""

    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks = chunks
        self.k1, self.b = k1, b
        self.docs = [Counter(tokenize(f"{c.section} {c.text}")) for c in chunks]
        self.lengths = [sum(d.values()) for d in self.docs]
        self.avg_len = sum(self.lengths) / max(len(self.lengths), 1)
        df: Counter = Counter()
        for d in self.docs:
            df.update(d.keys())
        n = len(chunks)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def search(self, query: str, k: int = 3) -> list[Hit]:
        q = tokenize(query)
        scores = []
        for i, (doc, length) in enumerate(zip(self.docs, self.lengths)):
            s = 0.0
            for t in q:
                tf = doc.get(t, 0)
                if tf:
                    norm = tf + self.k1 * (1 - self.b + self.b * length / self.avg_len)
                    s += self.idf[t] * tf * (self.k1 + 1) / norm
            if s > 0:
                scores.append((s, i))
        scores.sort(reverse=True)
        return [Hit(self.chunks[i], s) for s, i in scores[:k]]


def format_context(hits: list[Hit]) -> str:
    """Retrieved chunks as numbered, source-tagged passages for a prompt."""
    return "\n\n".join(f"[{h.chunk.source}] {h.chunk.section}\n{h.chunk.text}" for h in hits)
