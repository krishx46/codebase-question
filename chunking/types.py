from __future__ import annotations

from dataclasses import dataclass

LANGUAGES = {
    ".py": "python", ".js": "javascript", ".jsx": "javascript",
    ".ts": "typescript", ".tsx": "tsx",
}


@dataclass
class CodeChunk:
    text: str
    start_line: int
    end_line: int
    symbol_name: str
    symbol_type: str
    class_name: str = ""
    docstring: str = ""
    imports: str = ""
    calls: str = ""
