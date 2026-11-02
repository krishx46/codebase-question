from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from urllib.parse import quote

import streamlit as st

from answer import answer_question
from code_graph import render_code_explorer
from index.vector import VectorIndex
from ingest.repository import clone_repository

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"


def repo_key(url: str) -> str:
    return hashlib.sha1(url.strip().lower().encode()).hexdigest()[:12]


@st.cache_resource(show_spinner="Loading embedding and retrieval models…")
def load_index(path: str):
    return VectorIndex(Path(path))


def linked_answer(answer: str, repo_url: str) -> str:
    base = repo_url.rstrip("/").removesuffix(".git")
    pattern = re.compile(r"([\w./-]+\.[A-Za-z0-9]+):(\d+)-(\d+)")

    def replace(match):
        path, start, end = match.groups()
        link = f"{base}/blob/HEAD/{quote(path)}#L{start}-L{end}"
        return f"[{path}:{start}-{end}]({link})"

    return pattern.sub(replace, answer)


def render_chat(index: VectorIndex, repo_url: str):
    st.subheader("Ask about this codebase")
    if "messages" not in st.session_state:
        st.session_state.messages = []
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    pending_question = st.session_state.pop("pending_graph_question", None)
    question = st.chat_input("Ask how a feature works, where something is handled, or who calls a function…")
    if pending_question:
        question = pending_question
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Searching code and generating an answer…"):
                answer, hits = answer_question(index, question)
            st.markdown(linked_answer(answer, repo_url))
            with st.expander(f"Retrieved evidence ({len(hits)} chunks)"):
                for hit in hits:
                    meta = hit["metadata"]
                    st.markdown(f"**{meta['file_path']}:{meta['start_line']}-{meta['end_line']}** · {meta.get('symbol_name') or 'code block'}")
                    st.code(hit["text"], language=meta.get("language", "text"))
        st.session_state.messages.append({"role": "assistant", "content": linked_answer(answer, repo_url)})


def main():
    st.set_page_config(page_title="Codebase Q&A", page_icon="🔎", layout="wide")
    st.title("Codebase Q&A")
    st.caption("Explore a repository map and ask questions grounded in cited source code.")
    with st.sidebar:
        st.header("Repository")
        repo_url = st.text_input("GitHub repository URL", placeholder="https://github.com/owner/repo")
        key = repo_key(repo_url) if repo_url.strip() else ""
        repo_dir = DATA / "repositories" / key if key else DATA / "repositories" / "empty"
        db_dir = DATA / "indexes" / key if key else DATA / "indexes" / "empty"
        do_index = st.button("Index / update repository", type="primary", disabled=not repo_url.strip())
        if do_index:
            progress = st.progress(0.0, text="Cloning or refreshing repository…")
            try:
                clone_repository(repo_url.strip(), repo_dir)
                vector_index = load_index(str(db_dir))

                def update_progress(done, total, relative):
                    ratio = done / max(total, 1)
                    progress.progress(ratio, text=f"Indexing {relative} ({done}/{total})")

                count = vector_index.index_repository(repo_dir, progress_callback=update_progress)
                st.session_state.active_key = key
                st.session_state.active_url = repo_url.strip()
                st.session_state.messages = []
                progress.progress(1.0, text=f"Ready. Indexed {count} changed chunks.")
            except Exception as error:
                st.error(str(error))

        if repo_url.strip() and db_dir.exists():
            if st.button("Open existing index"):
                st.session_state.active_key = key
                st.session_state.active_url = repo_url.strip()
                st.session_state.messages = []
        st.caption(f"Ollama model: {os.environ.get('OLLAMA_MODEL', 'qwen3:8b')}")

    active_key = st.session_state.get("active_key")
    if active_key:
        active_repo = DATA / "repositories" / active_key
        active_db = DATA / "indexes" / active_key
        active_url = st.session_state.get("active_url", repo_url)
        vector_index = load_index(str(active_db))
        chat_col, map_col = st.columns([1.05, 0.95], gap="large")
        with chat_col:
            render_chat(vector_index, active_url)
        with map_col:
            render_code_explorer(vector_index, active_repo, active_url, active_key)
    else:
        st.info("Paste a public GitHub repository URL in the sidebar and click Index to get started.")


if __name__ == "__main__":
    main()
