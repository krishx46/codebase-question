from pathlib import Path

import pytest

from chunking.ast_chunks import chunk_source
from chunking.ast_chunks import get_parser

AST_AVAILABLE = get_parser is not None


@pytest.mark.skipif(not AST_AVAILABLE, reason="Tree-sitter native parser is blocked or unavailable")
def test_python_functions_have_line_ranges_and_docstrings():
    source = '"""module"""\n\ndef greet(name):\n    """Say hello."""\n    return "hi " + name\n'
    chunks = chunk_source(Path("hello.py"), source)
    assert len(chunks) == 1
    assert chunks[0].symbol_name == "greet"
    assert chunks[0].symbol_type == "function"
    assert chunks[0].start_line == 3
    assert chunks[0].end_line == 5
    assert chunks[0].docstring == "Say hello."


@pytest.mark.skipif(not AST_AVAILABLE, reason="Tree-sitter native parser is blocked or unavailable")
def test_class_methods_include_class_context():
    source = "class Cart:\n    def total(self):\n        return 3\n"
    chunks = chunk_source(Path("cart.py"), source)
    assert len(chunks) == 1
    assert chunks[0].symbol_name == "total"
    assert chunks[0].symbol_type == "method"
    assert chunks[0].class_name == "Cart"
    assert "class Cart" in chunks[0].text


@pytest.mark.skipif(not AST_AVAILABLE, reason="Tree-sitter native parser is blocked or unavailable")
def test_javascript_functions_are_extracted():
    source = "export function charge(amount) {\n  return amount;\n}\n"
    chunks = chunk_source(Path("payment.js"), source)
    assert chunks[0].symbol_name == "charge"
    assert chunks[0].start_line == 1
    assert chunks[0].end_line == 3


@pytest.mark.skipif(not AST_AVAILABLE, reason="Tree-sitter native parser is blocked or unavailable")
def test_imports_are_retained_as_chunk_context():
    source = "from mailer import send\n\ndef notify(address):\n    send(address)\n"
    chunk = chunk_source(Path("notify.py"), source)[0]
    assert "from mailer import send" in chunk.imports
    assert "# Imports:" in chunk.text


@pytest.mark.skipif(not AST_AVAILABLE, reason="Tree-sitter native parser is blocked or unavailable")
def test_oversized_function_is_split_with_overlap():
    source = "def long_task():\n" + "    process_record()\n" * 700
    chunks = chunk_source(Path("long_task.py"), source)
    assert len(chunks) > 1
    assert all(chunk.symbol_name == "long_task" for chunk in chunks)
    assert any(left.end_line >= right.start_line for left, right in zip(chunks, chunks[1:]))


def test_unsupported_language_uses_fallback_chunks():
    chunks = chunk_source(Path("notes.md"), "one\ntwo\n")
    assert chunks
    assert chunks[0].symbol_type == "block"
