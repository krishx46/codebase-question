from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

from chunking.ast_chunks import chunk_source


def main() -> None:
    payload = json.load(sys.stdin)
    path = Path(payload["path"])
    chunks = chunk_source(path, payload["text"])
    json.dump([asdict(chunk) for chunk in chunks], sys.stdout)


if __name__ == "__main__":
    main()
