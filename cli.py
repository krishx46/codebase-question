from __future__ import annotations

from pathlib import Path

import typer

from answer import answer_question
from index.vector import VectorIndex
from ingest.repository import clone_repository

app = typer.Typer(help="Index a GitHub repository and ask code questions.")


@app.command()
def index(
    repo_url: str,
    repo_dir: Path = Path("data/repository"),
    db_dir: Path = Path("data/chroma"),
):
    """Shallow-clone and index a public GitHub repository."""
    root = clone_repository(repo_url, repo_dir)
    vector_index = VectorIndex(db_dir)
    count = vector_index.index_repository(root)
    typer.echo(f"Indexed {count} chunks from {repo_url}")


@app.command()
def ask(question: str, db_dir: Path = Path("data/chroma"), top_k: int = 8):
    """Ask a question about the already indexed repository."""
    vector_index = VectorIndex(db_dir)
    answer, hits = answer_question(vector_index, question, top_k=top_k)
    typer.echo(answer)
    typer.echo("\nRetrieved chunks:")
    for hit in hits:
        meta = hit["metadata"]
        score = hit.get("rerank_score")
        score_label = f"rerank={score:.3f}" if score is not None else "rerank unavailable"
        typer.echo(f"- {meta['file_path']}:{meta['start_line']}-{meta['end_line']} ({score_label})")


if __name__ == "__main__":
    app()
