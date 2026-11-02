from __future__ import annotations

import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def tokenize(text: str) -> list[str]:
    """Keep identifier-like terms intact while splitting camelCase and snake_case."""
    terms = re.findall(r"[A-Za-z]+|[0-9]+", text)
    expanded = []
    for term in terms:
        expanded.extend(re.findall(r"[A-Z]+(?=[A-Z][a-z]|$)|[A-Z]?[a-z]+|[0-9]+", term))
    return [term.lower() for term in expanded if term]


class HybridRetriever:
    def __init__(self, db_path: Path, reranker_model: str = RERANKER_MODEL):
        self.corpus_path = db_path / "bm25_corpus.json"
        self.reranker_model = reranker_model
        self._reranker = None
        self._corpus: list[dict] | None = None
        self._bm25: BM25Okapi | None = None

    def _load_bm25(self) -> None:
        if self._corpus is not None:
            return
        if not self.corpus_path.exists():
            self._corpus = []
        else:
            self._corpus = json.loads(self.corpus_path.read_text(encoding="utf-8"))
        tokenized = [tokenize(item["text"] + " " + item["metadata"].get("file_path", ""))
                     for item in self._corpus]
        self._bm25 = BM25Okapi(tokenized) if tokenized and any(tokenized) else None

    def retrieve(self, question: str, semantic_hits: list[dict], top_k: int = 8) -> list[dict]:
        """Fuse semantic and BM25 rankings with RRF, then cross-encoder rerank."""
        self._load_bm25()
        by_id = {hit["id"]: hit for hit in semantic_hits}
        scores: dict[str, float] = {}
        for rank, hit in enumerate(semantic_hits, start=1):
            scores[hit["id"]] = scores.get(hit["id"], 0.0) + 1.0 / (60 + rank)

        if self._bm25 and self._corpus:
            bm25_scores = self._bm25.get_scores(tokenize(question))
            bm25_order = sorted(range(len(bm25_scores)), key=bm25_scores.__getitem__, reverse=True)[:30]
            for rank, corpus_idx in enumerate(bm25_order, start=1):
                item = self._corpus[corpus_idx]
                chunk_id = item["id"]
                if bm25_scores[corpus_idx] <= 0:
                    continue
                by_id.setdefault(chunk_id, item)
                scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (60 + rank)

        candidates = sorted(by_id.values(), key=lambda item: scores.get(item["id"], 0), reverse=True)[:30]
        if not candidates:
            return []
        if self._reranker is None:
            from sentence_transformers import CrossEncoder

            self._reranker = CrossEncoder(self.reranker_model)
        pairs = [(question, item["text"]) for item in candidates]
        rerank_scores = self._reranker.predict(pairs)
        ranked = sorted(zip(candidates, rerank_scores), key=lambda pair: float(pair[1]), reverse=True)
        return [
            {"distance": None, **item, "rerank_score": float(score)}
            for item, score in ranked[:top_k]
        ]
