from __future__ import annotations


def fixed_chunks(text: str, size: int = 1800, overlap: int = 250):
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError("Require size > overlap >= 0")
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        yield start, end, text[start:end]
        if end == len(text):
            break
        start = end - overlap


def line_for_offset(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1
