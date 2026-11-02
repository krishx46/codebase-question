from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path

from chunking.fixed import fixed_chunks, line_for_offset
from chunking.types import CodeChunk, LANGUAGES

logger = logging.getLogger(__name__)


def _fallback_chunks(text: str) -> list[CodeChunk]:
    return [CodeChunk(part, line_for_offset(text, start), line_for_offset(text, end), "", "block")
            for start, end, part in fixed_chunks(text)]


def safe_chunk_source(path: Path, text: str) -> list[CodeChunk]:
    """Run native Tree-sitter parsing out of process so a parser crash is contained."""
    if path.suffix.lower() not in LANGUAGES:
        return _fallback_chunks(text)

    project_root = Path(__file__).resolve().parents[1]
    payload = json.dumps({"path": str(path), "text": text})
    try:
        result = subprocess.run(
            [sys.executable, "-m", "chunking.worker"],
            input=payload,
            text=True,
            capture_output=True,
            cwd=project_root,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        logger.warning("AST worker failed for %s (%s); using fixed-size chunks", path, error)
        return _fallback_chunks(text)

    if result.returncode != 0:
        logger.warning(
            "AST worker exited with code %s for %s; using fixed-size chunks. %s",
            result.returncode,
            path,
            result.stderr[-2000:],
        )
        return _fallback_chunks(text)

    try:
        return [CodeChunk(**item) for item in json.loads(result.stdout)]
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        logger.warning("Invalid AST worker output for %s (%s); using fixed-size chunks", path, error)
        return _fallback_chunks(text)
