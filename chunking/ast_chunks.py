from __future__ import annotations

from pathlib import Path

try:
    from tree_sitter_language_pack import get_parser
except (ImportError, OSError):
    # Keep indexing usable when a host policy blocks the parser's native DLL.
    get_parser = None

from chunking.fixed import fixed_chunks, line_for_offset
from chunking.types import CodeChunk, LANGUAGES
FUNCTION_TYPES = {
    "function_definition", "function_declaration", "generator_function",
    "generator_function_declaration", "method_definition", "arrow_function",
}
CLASS_TYPES = {"class_definition", "class_declaration"}
MAX_CHUNK_CHARS = 5000
LONG_OVERLAP_LINES = 12


def _node_text(source: bytes, node) -> str:
    return source[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _name_for(node, source: bytes, fallback: str) -> str:
    name = node.child_by_field_name("name")
    if name:
        return _node_text(source, name)
    # Arrow functions and assigned function expressions use the variable name.
    parent = node.parent
    if parent and parent.type in {"variable_declarator", "assignment_expression"}:
        left = parent.child_by_field_name("name") or parent.child_by_field_name("left")
        if left:
            return _node_text(source, left)
    return fallback


def _docstring(node, source: bytes, language: str) -> str:
    if language != "python":
        return ""
    body = node.child_by_field_name("body")
    if body and body.named_children and body.named_children[0].type == "expression_statement":
        first = body.named_children[0].named_children
        if first and first[0].type == "string":
            return _node_text(source, first[0]).strip("\"'\n ")
    return ""


def _calls(node, source: bytes) -> str:
    found: set[str] = set()
    stack = [node]
    while stack:
        current = stack.pop()
        if current.type in {"call", "call_expression"}:
            function = current.child_by_field_name("function")
            if function:
                label = _node_text(source, function)
                # Keep the terminal identifier for a useful lightweight call map.
                found.add(label.rsplit(".", 1)[-1].split("(", 1)[0])
        stack.extend(current.named_children)
    return ",".join(sorted(found))


def _split_oversize(text: str, start_line: int, max_chars: int = MAX_CHUNK_CHARS):
    lines = text.splitlines(keepends=True)
    if len(text) <= max_chars:
        return [(text, start_line, start_line + max(0, len(lines) - 1))]
    parts = []
    cursor = 0
    while cursor < len(lines):
        end, length = cursor, 0
        while end < len(lines) and length + len(lines[end]) <= max_chars:
            length += len(lines[end])
            end += 1
        if end == cursor:
            end += 1
        piece = "".join(lines[cursor:end])
        parts.append((piece, start_line + cursor, start_line + end - 1))
        if end == len(lines):
            break
        cursor = max(cursor + 1, end - LONG_OVERLAP_LINES)
    return parts


def chunk_source(path: Path, text: str) -> list[CodeChunk]:
    """Extract functions/methods and class signatures; fallback to fixed chunks."""
    language = LANGUAGES.get(path.suffix.lower())
    if not language or not text.strip() or get_parser is None:
        return _fallback_chunks(text)
    try:
        tree = get_parser(language).parse(text.encode("utf-8"))
    except Exception:
        return _fallback_chunks(text)

    source = text.encode("utf-8")
    root = tree.root_node
    imports = [
        _node_text(source, node).strip()
        for node in root.named_children
        if node.type in {"import_statement", "import_from_statement", "import_declaration", "lexical_declaration"}
        and "import" in _node_text(source, node)[:20]
    ]
    import_text = "\n".join(imports)
    chunks: list[CodeChunk] = []
    seen: set[tuple[int, int]] = set()

    def add(node, symbol_type: str, name: str, class_name: str = "", class_context: str = ""):
        key = (node.start_byte, node.end_byte)
        if key in seen:
            return
        seen.add(key)
        body = _node_text(source, node)
        prefix = f"# Symbol: {class_name + '.' if class_name else ''}{name} ({symbol_type})\n"
        if class_context:
            prefix += f"# Class context: {class_context}\n"
        if import_text:
            prefix += f"# Imports: {import_text}\n"
        doc = _docstring(node, source, language)
        calls = _calls(node, source)
        content = prefix + body
        for part, first, last in _split_oversize(body, node.start_point.row + 1):
            part_prefix = prefix if first == node.start_point.row + 1 else (
                f"# Symbol continuation: {class_name + '.' if class_name else ''}{name}\n"
                + (f"# Class context: {class_context}\n" if class_context else "")
                + (f"# Imports: {import_text}\n" if import_text else "")
            )
            chunks.append(CodeChunk(part_prefix + part, first, last, name, symbol_type,
                                    class_name, doc, import_text, calls))

    def visit(node, current_class: str = "", class_context: str = ""):
        if node.type in CLASS_TYPES:
            name = _name_for(node, source, f"class_at_{node.start_point.row + 1}")
            body = node.child_by_field_name("body")
            signature = _node_text(source, node).splitlines()[0]
            # Python/JS class-level methods are independent chunks with class signature context.
            if body:
                for child in body.named_children:
                    if child.type in FUNCTION_TYPES:
                        add(child, "method", _name_for(child, source, f"method_at_{child.start_point.row + 1}"), name, signature)
                    else:
                        visit(child, name, signature)
            return
        if node.type in FUNCTION_TYPES:
            kind = "method" if current_class else "function"
            add(node, kind, _name_for(node, source, f"function_at_{node.start_point.row + 1}"), current_class, class_context)
            return
        for child in node.named_children:
            visit(child, current_class, class_context)

    for child in root.named_children:
        visit(child)
    return chunks or _fallback_chunks(text)


def _fallback_chunks(text: str) -> list[CodeChunk]:
    return [CodeChunk(part, line_for_offset(text, start), line_for_offset(text, end),
                      "", "block") for start, end, part in fixed_chunks(text)]
