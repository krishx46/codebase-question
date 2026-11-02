from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from urllib.parse import urlparse

SUPPORTED_EXTENSIONS = {
    ".py", ".js", ".jsx", ".ts", ".tsx", ".java", ".go", ".rs",
    ".rb", ".php", ".c", ".h", ".cpp", ".hpp", ".cs", ".swift",
    ".kt", ".scala", ".sh", ".sql", ".html", ".css", ".scss",
    ".json", ".yaml", ".yml", ".toml", ".md",
}
SKIP_DIRS = {
    ".git", "node_modules", "build", "dist", "target", "vendor",
    "__pycache__", ".next", ".venv", "venv", "coverage", "chroma_db",
}
SKIP_NAMES = {
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml", "poetry.lock",
    "pdm.lock", "uv.lock", "Cargo.lock", "Gemfile.lock", "composer.lock",
}
MAX_FILE_BYTES = 1_000_000


def validate_repo_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.netloc.lower() != "github.com":
        raise ValueError("Repository URL must be an HTTPS URL on github.com")
    if len([part for part in parsed.path.strip("/").split("/") if part]) < 2:
        raise ValueError("Repository URL must include an owner and repository name")


def clone_repository(url: str, destination: Path) -> Path:
    """Shallow clone a public GitHub repository into destination."""
    validate_repo_url(url)
    destination = destination.resolve()
    if destination.exists() and any(destination.iterdir()):
        if not (destination / ".git").exists():
            raise FileExistsError(f"Destination is not an existing Git checkout: {destination}")
        remote = subprocess.run(
            ["git", "-C", str(destination), "remote", "get-url", "origin"],
            capture_output=True, text=True, check=False,
        )
        normalize = lambda value: value.rstrip("/").removesuffix(".git").lower()
        if remote.returncode or normalize(remote.stdout.strip()) != normalize(url):
            raise ValueError("The existing checkout's origin does not match the requested URL")
        result = subprocess.run(
            ["git", "-C", str(destination), "fetch", "--depth", "1", "origin"],
            capture_output=True, text=True, check=False,
        )
        if result.returncode:
            raise RuntimeError(f"git fetch failed: {result.stderr.strip()}")
        result = subprocess.run(
            ["git", "-C", str(destination), "reset", "--hard", "FETCH_HEAD"],
            capture_output=True, text=True, check=False,
        )
        if result.returncode:
            raise RuntimeError(f"git checkout refresh failed: {result.stderr.strip()}")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["git", "clone", "--depth", "1", "--", url, str(destination)],
        capture_output=True, text=True, check=False,
    )
    if result.returncode:
        raise RuntimeError(f"git clone failed: {result.stderr.strip()}")
    return destination


def iter_source_files(root: Path):
    """Yield supported, reasonably sized text source files under root."""
    root = root.resolve()
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if any(part.lower() in SKIP_DIRS for part in relative.parts[:-1]):
            continue
        if path.name in SKIP_NAMES or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        if path.name.endswith((".min.js", ".min.css", ".generated.py", ".g.py")):
            continue
        try:
            if path.stat().st_size > MAX_FILE_BYTES:
                continue
            # Reject binary and invalid UTF-8 files early.
            path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            continue
        yield path


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
