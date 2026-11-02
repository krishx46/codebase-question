# Evaluation results

## Air-cargo repository

- Repository revision: `8283e3d1bd400ed99f0b53c20adab3ed2160e9b8`
- Labeled questions: 40
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Answer model: `qwen3:8b`

| Configuration | Recall@5 | Recall@10 | MRR |
|---|---:|---:|---:|
| Naive fixed-size + vector | 0.850 | 0.925 | 0.625 |
| AST + vector | 0.825 | 0.925 | 0.627 |
| AST + hybrid | 0.900 | 0.925 | 0.669 |
| AST + hybrid + reranker | **0.900** | **0.950** | **0.720** |

The hybrid setup with reranking ranked relevant code highest overall. Compared with AST + hybrid, it improved MRR by 0.051 and Recall@10 by 2.5 percentage points; Recall@5 was unchanged.

## Answer review

A code-level review of the first 12 generated answers for AST + hybrid + reranker scored faithfulness at **3.75/5**. Eight answers scored 4 or 5. Two answers were unrelated to their questions, and several other answers used broad or imprecise citations. This was a small, non-random sample reviewed by the project assistant, not a blinded human study.

## Limitations

- This is a single-repository result with one manually assigned target file and symbol per question.
- The Tree-sitter worker crashed on `agents/mcp_client/agent.py` and `python/generate_all.py`; fixed-size fallback chunks were used for those files in AST configurations.
- The answer review exposed failures that should be addressed before making broad claims about answer reliability.
- Results are preliminary and may change with a different repository, question set, or model version.

To reproduce retrieval metrics, follow the command in the [README evaluation section](../README.md#air-cargo-repository-benchmark). The full evidence and review sheets are generated locally under `eval/results/` and are excluded from Git because they contain retrieved source snippets.
