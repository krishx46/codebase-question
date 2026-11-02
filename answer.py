from __future__ import annotations

import os

from ollama import Client
from index.vector import VectorIndex


def answer_question(index: VectorIndex, question: str, top_k: int = 8) -> tuple[str, list[dict]]:
    hits = index.search(question, top_k=top_k)
    # Never call the model without retrieved code context.
    if not hits:
        return "I couldn't find this in the indexed code.", []

    context = "\n\n".join(
        f"[{hit['metadata']['file_path']}:{hit['metadata']['start_line']}-"
        f"{hit['metadata']['end_line']}]\n{hit['text']}"
        for hit in hits
    )
    client = Client(host=os.environ.get("OLLAMA_HOST", "http://localhost:11434"))
    response = client.chat(
        model=os.environ.get("OLLAMA_MODEL", "qwen3:8b"),
        messages=[
            {"role": "system", "content": (
            "Answer only from the provided repository code. Cite every factual claim "
            "with the exact file path and line range shown in the context. Do not "
            "invent citations. If the context is insufficient, say exactly: "
            "I couldn't find this in the indexed code."
            )},
            {"role": "user", "content": f"Question: {question}\n\nRetrieved code:\n{context}"},
        ],
    )
    return response.message.content, hits
