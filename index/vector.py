from __future__ import annotations

import hashlib
import json
from pathlib import Path

from chunking.safe import safe_chunk_source
from ingest.repository import file_hash, iter_source_files
from retrieval.hybrid import HybridRetriever

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _chunk_id(relative_path: str, start: int) -> str:
    return hashlib.sha1(f"{relative_path}:{start}".encode()).hexdigest()


class VectorIndex:
    def __init__(self, db_path: Path, model_name: str = DEFAULT_MODEL):
        import chromadb
        from sentence_transformers import SentenceTransformer

        self.db_path = db_path
        db_path.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(db_path))
        self.collection = self.client.get_or_create_collection(
            "code_chunks", metadata={"hnsw:space": "cosine"}
        )
        self.model = SentenceTransformer(model_name)
        self.retriever = HybridRetriever(db_path)

    def index_repository(self, root: Path, progress_callback=None) -> int:
        count = 0
        root = root.resolve()
        manifest_path = self.db_path / "file_manifest.json"
        old_manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
        files = {path.relative_to(root).as_posix(): path for path in iter_source_files(root)}
        new_manifest = {relative: file_hash(path) for relative, path in files.items()}

        # Remove deleted files and replace changed files so old chunks never linger.
        for relative in set(old_manifest) - set(new_manifest):
            self.collection.delete(where={"file_path": relative})
        changed = {relative for relative, digest in new_manifest.items()
                   if old_manifest.get(relative) != digest}
        for relative in changed:
            self.collection.delete(where={"file_path": relative})

        for file_number, relative in enumerate(sorted(changed), start=1):
            path = files[relative]
            text = path.read_text(encoding="utf-8")
            chunk_ids, documents, metadatas = [], [], []
            for chunk_number, chunk in enumerate(safe_chunk_source(path, text)):
                start_line = chunk.start_line
                end_line = chunk.end_line
                chunk_id = _chunk_id(relative, f"{start_line}:{chunk.symbol_name}:{chunk_number}")
                chunk_ids.append(chunk_id)
                documents.append(chunk.text)
                metadatas.append({
                        "file_path": relative,
                        "start_line": start_line,
                        "end_line": end_line,
                        "language": path.suffix.lower().lstrip("."),
                        "symbol_name": chunk.symbol_name,
                        "symbol_type": chunk.symbol_type,
                        "class_name": chunk.class_name,
                        "docstring": chunk.docstring,
                        "imports": chunk.imports,
                        "calls": chunk.calls,
                    })
            if documents:
                embeddings = self.model.encode(documents).tolist()
                self.collection.upsert(ids=chunk_ids, documents=documents,
                                       embeddings=embeddings, metadatas=metadatas)
                count += len(documents)
            if progress_callback:
                progress_callback(file_number, len(changed), relative)
        manifest_path.write_text(json.dumps(new_manifest, indent=2), encoding="utf-8")
        self._save_bm25_corpus()
        self.retriever._corpus = None
        self.retriever._bm25 = None
        return count

    def _save_bm25_corpus(self) -> None:
        """Persist the same Chroma documents and metadata for BM25 search."""
        rows = self.collection.get(include=["documents", "metadatas"])
        corpus = [
            {"id": chunk_id, "text": doc, "metadata": metadata}
            for chunk_id, doc, metadata in zip(rows["ids"], rows["documents"], rows["metadatas"])
        ]
        (self.db_path / "bm25_corpus.json").write_text(
            json.dumps(corpus, ensure_ascii=False), encoding="utf-8"
        )

    def search(self, question: str, top_k: int = 8):
        if self.collection.count() == 0:
            return []
        semantic_count = min(30, self.collection.count())
        result = self.collection.query(
            query_embeddings=[self.model.encode(question).tolist()],
            n_results=semantic_count,
            include=["documents", "metadatas", "distances"],
        )
        semantic = [
            {"id": chunk_id, "text": doc, "metadata": metadata, "distance": distance}
            for chunk_id, doc, metadata, distance in zip(
                result["ids"][0], result["documents"][0], result["metadatas"][0],
                result["distances"][0],
            )
        ]
        return self.retriever.retrieve(question, semantic, top_k=top_k)
