from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
from pathlib import Path

from ollama import Client
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

from chunking.fixed import fixed_chunks, line_for_offset
from chunking.safe import safe_chunk_source
from retrieval.hybrid import RERANKER_MODEL, tokenize

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPO = Path(__file__).parent / "test_repo"
DEFAULT_QUESTIONS = Path(__file__).parent / "questions.jsonl"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CONFIGURATIONS = (
    ("Naive fixed-size + vector", "naive", "vector", False),
    ("AST + vector", "ast", "vector", False),
    ("AST + hybrid", "ast", "hybrid", False),
    ("AST + hybrid + reranker", "ast", "hybrid", True),
)


def load_chunks(repo: Path, ast: bool) -> list[dict]:
    chunks = []
    ignored = {".git", "node_modules", "dist", "build", "__pycache__", ".venv"}
    extensions = {".py", ".js", ".jsx", ".ts", ".tsx"}
    for path in sorted(repo.rglob("*")):
        if not path.is_file() or any(part in ignored for part in path.parts):
            continue
        if path.suffix.lower() not in extensions:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            continue
        relative = path.relative_to(repo).as_posix()
        if ast:
            records = safe_chunk_source(path, text)
            chunks.extend({"file": relative, "symbol": c.symbol_name, "start_line": c.start_line,
                           "end_line": c.end_line, "text": c.text} for c in records)
        else:
            chunks.extend({"file": relative, "symbol": "", "start_line": line_for_offset(text, start),
                           "end_line": line_for_offset(text, end), "text": chunk}
                          for start, end, chunk in fixed_chunks(text))
    if not chunks:
        raise ValueError(f"No supported source files found under {repo}")
    return chunks


def is_relevant(chunk: dict, question: dict) -> bool:
    return chunk["file"] == question["file"] and (
        chunk["symbol"] == question["symbol"] or question["symbol"] in chunk["text"]
    )


def git_commit(repo: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True, timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Evaluate code retrieval configurations on a labeled repository.")
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO,
                        help="Repository directory (defaults to the bundled demo repo).")
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS,
                        help="JSONL questions with expected file and symbol labels.")
    parser.add_argument("--embedding-model", default=EMBED_MODEL)
    parser.add_argument("--reranker-model", default=RERANKER_MODEL)
    parser.add_argument("--output-dir", type=Path, default=Path("eval/results"))
    parser.add_argument("--generate-answers", action="store_true",
                        help="Generate a bounded set of answers with Ollama for manual review.")
    parser.add_argument("--answer-limit", type=int, default=12,
                        help="Maximum number of Ollama answers to generate (default: 12).")
    parser.add_argument("--answer-config", default="AST + hybrid + reranker",
                        choices=[c[0] for c in CONFIGURATIONS],
                        help="Retrieval configuration used for generated answer samples.")
    args = parser.parse_args()
    repo = args.repo.resolve()
    question_path = args.questions.resolve()
    output_dir = args.output_dir.resolve()
    if not repo.is_dir():
        parser.error(f"Repository directory does not exist: {repo}")
    if args.answer_limit < 0:
        parser.error("--answer-limit must be zero or greater")
    output_dir.mkdir(parents=True, exist_ok=True)
    questions = [json.loads(row) for row in question_path.read_text(encoding="utf-8").splitlines() if row.strip()]
    if not questions:
        parser.error(f"No questions found in {question_path}")

    naive = load_chunks(repo, ast=False)
    ast = load_chunks(repo, ast=True)
    encoder = SentenceTransformer(args.embedding_model)
    reranker = CrossEncoder(args.reranker_model)
    ollama_client = Client(host=os.environ.get("OLLAMA_HOST", "http://localhost:11434")) if args.generate_answers else None
    answer_model = os.environ.get("OLLAMA_MODEL", "qwen3:8b")
    answer_budget = min(args.answer_limit, len(questions)) if args.generate_answers else 0
    answer_config = args.answer_config
    metrics = []
    manual_rows = []
    candidate_sets: dict[str, list[list[int]]] = {}

    for title, chunk_kind, retrieval_mode, use_reranker in CONFIGURATIONS:
        chunks = naive if chunk_kind == "naive" else ast
        vectors = encoder.encode([item["text"] for item in chunks], normalize_embeddings=True)
        tokenized = [tokenize(item["text"] + " " + item["file"]) for item in chunks]
        bm25 = BM25Okapi(tokenized)
        reciprocal_ranks, recall5, recall10 = [], 0, 0
        candidate_sets[title] = []
        for question_number, question in enumerate(questions):
            query = encoder.encode(question["question"], normalize_embeddings=True)
            vector_order = list(vectors @ query)
            vector_ranks = sorted(range(len(chunks)), key=vector_order.__getitem__, reverse=True)[:30]
            if retrieval_mode == "vector":
                candidates = vector_ranks
            else:
                lexical_scores = bm25.get_scores(tokenize(question["question"]))
                lexical_ranks = sorted(range(len(chunks)), key=lexical_scores.__getitem__, reverse=True)[:30]
                fused: dict[int, float] = {}
                for ranking in (vector_ranks, lexical_ranks):
                    for rank, idx in enumerate(ranking, start=1):
                        fused[idx] = fused.get(idx, 0.0) + 1 / (60 + rank)
                candidates = sorted(fused, key=fused.__getitem__, reverse=True)[:30]
            if use_reranker:
                scores = reranker.predict([(question["question"], chunks[idx]["text"]) for idx in candidates])
                candidates = [idx for idx, _ in sorted(zip(candidates, scores), key=lambda pair: float(pair[1]), reverse=True)]
            candidate_sets[title].append(candidates)

            relevant_ranks = [rank for rank, idx in enumerate(candidates, 1) if is_relevant(chunks[idx], question)]
            first = relevant_ranks[0] if relevant_ranks else None
            reciprocal_ranks.append(1 / first if first else 0)
            recall5 += int(any(rank <= 5 for rank in relevant_ranks))
            recall10 += int(any(rank <= 10 for rank in relevant_ranks))
            evidence = "\n\n".join(
                f"[{chunks[idx]['file']}:{chunks[idx]['start_line']}-{chunks[idx]['end_line']} :: "
                f"{chunks[idx]['symbol']}]\n{chunks[idx]['text']}" for idx in candidates[:5]
            )
            manual_rows.append({
                "configuration": title,
                "question": question["question"],
                "expected_file": question["file"],
                "expected_symbol": question["symbol"],
                "retrieved_evidence": evidence,
                "answer": "",
                "answer_generated": "no",
                "faithfulness_score_1_to_5": "",
                "reviewer_notes": "",
            })
        total = len(questions)
        metrics.append({"configuration": title, "recall@5": recall5 / total,
                        "recall@10": recall10 / total,
                        "MRR": sum(reciprocal_ranks) / total,
                        "manual faithfulness": "pending review"})
        print(f"Finished retrieval: {title}", flush=True)

    # Persist retrieval scores before the optional, slower local answer-generation stage.
    write_csv(output_dir / "metrics.csv", metrics)
    write_csv(output_dir / "manual_review.csv", manual_rows)
    metadata = {
        "repository": str(repo),
        "git_commit": git_commit(repo),
        "question_set": str(question_path),
        "question_count": len(questions),
        "embedding_model": args.embedding_model,
        "reranker_model": args.reranker_model,
        "ollama_model": answer_model if args.generate_answers else None,
        "answer_config": answer_config if args.generate_answers else None,
        "answer_sampling": f"first {answer_budget} questions in the JSONL set" if args.generate_answers else None,
        "generated_answers": 0,
    }
    (output_dir / "run_info.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Repository: {repo} (commit {metadata['git_commit'] or 'unknown'})")
    print(f"Questions: {len(questions)}; generated answers: {answer_budget}")
    print("| Configuration | Recall@5 | Recall@10 | MRR | Manual faithfulness |")
    print("|---|---:|---:|---:|---|")
    for row in metrics:
        print(f"| {row['configuration']} | {row['recall@5']:.3f} | {row['recall@10']:.3f} | {row['MRR']:.3f} | pending review |")
    if ollama_client and answer_budget:
        print(f"Starting {answer_budget} answer samples with {answer_model} ({answer_config})", flush=True)
        chunks = naive if answer_config == CONFIGURATIONS[0][0] else ast
        for question_number in range(answer_budget):
            question = questions[question_number]
            candidates = candidate_sets[answer_config][question_number]
            context = "\n\n".join(
                f"[{chunks[idx]['file']}:{chunks[idx]['start_line']}-{chunks[idx]['end_line']}]\n"
                f"{chunks[idx]['text']}" for idx in candidates[:8]
            )
            response = ollama_client.chat(
                model=answer_model,
                messages=[
                    {"role": "system", "content": "Answer briefly and only from the supplied code. Cite every factual claim with the exact file and line range. If unsupported, say you could not find it in the indexed code."},
                    {"role": "user", "content": f"Question: {question['question']}\n\nRetrieved code:\n{context}"},
                ],
                think=False,
                options={"num_predict": 384, "temperature": 0},
            )
            row = next(r for r in manual_rows if r["configuration"] == answer_config and r["question"] == question["question"])
            row["answer"] = response.message.content
            row["answer_generated"] = "yes"
            write_csv(output_dir / "manual_review.csv", manual_rows)
            metadata["generated_answers"] = question_number + 1
            (output_dir / "run_info.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            print(f"Generated answer {question_number + 1}/{answer_budget} with {answer_model}", flush=True)
    print(f"Results: {output_dir}")


if __name__ == "__main__":
    main()
