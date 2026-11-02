from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path
from urllib.parse import quote

import streamlit as st
from streamlit_agraph import Config, Edge, Node, agraph

PALETTE = ["#5B8DEF", "#A78BFA", "#34D399", "#FBBF24", "#F472B6", "#22D3EE", "#FB7185"]
MAX_GRAPH_NODES = 60


def build_code_map(index):
    rows = index.collection.get(include=["metadatas"])
    files: dict[str, set[str]] = {}
    symbols: dict[str, dict] = {}

    for metadata in rows.get("metadatas") or []:
        path = metadata.get("file_path", "")
        if not path:
            continue
        files.setdefault(path, set())
        name = metadata.get("symbol_name") or ""
        if not name:
            continue

        class_name = metadata.get("class_name") or ""
        identity = (path, class_name, name)
        node_id = "::".join(identity)
        record = symbols.setdefault(
            node_id,
            {
                "id": node_id,
                "file_path": path,
                "symbol_name": name,
                "class_name": class_name,
                "symbol_type": metadata.get("symbol_type") or "function",
                "start_line": metadata.get("start_line", 1),
                "end_line": metadata.get("end_line", 1),
                "docstring": metadata.get("docstring") or "",
                "calls": set(),
            },
        )
        record["start_line"] = min(record["start_line"], metadata.get("start_line", 1))
        record["end_line"] = max(record["end_line"], metadata.get("end_line", 1))
        record["calls"].update(filter(None, (metadata.get("calls") or "").split(",")))
        files[path].add(name)

    by_name: dict[str, list[str]] = defaultdict(list)
    for node_id, node in symbols.items():
        by_name[node["symbol_name"]].append(node_id)

    edges: list[dict] = []
    unresolved: list[dict] = []
    edge_pairs: set[tuple[str, str]] = set()
    for caller_id, caller in symbols.items():
        for called_name in sorted(caller["calls"]):
            targets = by_name.get(called_name, [])
            if not targets:
                unresolved.append({"source": caller_id, "name": called_name})
                continue
            for target_id in targets:
                pair = (caller_id, target_id)
                if pair in edge_pairs:
                    continue
                edge_pairs.add(pair)
                edges.append({
                    "id": f"{caller_id}=>{target_id}",
                    "source": caller_id,
                    "target": target_id,
                    "label": "possible call",
                })

    return files, symbols, edges, unresolved


def _node_label(node: dict) -> str:
    qualified = f"{node['class_name']}.{node['symbol_name']}" if node["class_name"] else node["symbol_name"]
    return f"{qualified} · {node['file_path']}:{node['start_line']}"


def _neighborhood(focus_id: str, symbols: dict[str, dict], edges: list[dict],
                  unresolved: list[dict], depth: int, include_external: bool):
    all_edges = list(edges)
    external_nodes: dict[str, dict] = {}
    if include_external:
        for item in unresolved:
            external_id = f"external::{item['name']}"
            external_nodes.setdefault(external_id, {
                "id": external_id,
                "symbol_name": item["name"],
                "file_path": "External or unresolved",
                "class_name": "",
                "symbol_type": "external",
                "start_line": 0,
                "end_line": 0,
                "docstring": "No indexed function with this name was found.",
                "calls": set(),
            })
            all_edges.append({
                "id": f"{item['source']}=>{external_id}",
                "source": item["source"],
                "target": external_id,
                "label": "unresolved call",
            })

    adjacency: dict[str, set[str]] = defaultdict(set)
    degree: dict[str, int] = defaultdict(int)
    for edge in all_edges:
        source, target = edge["source"], edge["target"]
        adjacency[source].add(target)
        adjacency[target].add(source)
        degree[source] += 1
        degree[target] += 1

    included = {focus_id}
    frontier = {focus_id}
    for _ in range(depth):
        candidates = set().union(*(adjacency[node_id] for node_id in frontier)) if frontier else set()
        candidates -= included
        remaining = MAX_GRAPH_NODES - len(included)
        if remaining <= 0:
            break
        def candidate_order(node_id: str):
            node = symbols[node_id] if node_id in symbols else external_nodes[node_id]
            return -degree[node_id], _node_label(node)

        ordered = sorted(candidates, key=candidate_order)
        frontier = set(ordered[:remaining])
        included.update(frontier)
        if len(included) >= MAX_GRAPH_NODES:
            break

    visible_nodes = {
        node_id: symbols[node_id] if node_id in symbols else external_nodes[node_id]
        for node_id in sorted(included)
    }
    visible_edges = [edge for edge in all_edges
                     if edge["source"] in included and edge["target"] in included]
    return visible_nodes, visible_edges, len(included) >= MAX_GRAPH_NODES


def _render_graph(symbols, edges, unresolved, repo_key: str, repo_dir: Path, repo_url: str):
    if not symbols:
        st.info("No function or method symbols were found in this repository index yet.")
        return

    folders = sorted({node["file_path"].split("/")[0] if "/" in node["file_path"] else "(root)"
                      for node in symbols.values()})
    folder = st.selectbox("Focus folder", ["All folders", *folders], key=f"graph_folder_{repo_key}")
    available = [node_id for node_id, node in symbols.items()
                 if folder == "All folders"
                 or (node["file_path"].split("/")[0] if "/" in node["file_path"] else "(root)") == folder]
    degree: dict[str, int] = defaultdict(int)
    for edge in edges:
        degree[edge["source"]] += 1
        degree[edge["target"]] += 1
    available.sort(key=lambda node_id: (-degree[node_id], _node_label(symbols[node_id]).lower()))
    focus_id = st.selectbox(
        "Find a function or method",
        available,
        index=0,
        format_func=lambda node_id: _node_label(symbols[node_id]),
        key=f"graph_focus_{repo_key}_{folder}",
    )
    control_col, external_col = st.columns([1, 2])
    with control_col:
        depth = st.selectbox("Connection depth", [1, 2], index=0, key=f"graph_depth_{repo_key}")
    with external_col:
        include_external = st.checkbox("Show unresolved calls", value=False, key=f"graph_external_{repo_key}")

    visible_nodes, visible_edges, capped = _neighborhood(
        focus_id, symbols, edges, unresolved, depth, include_external
    )
    if capped:
        st.caption(f"Showing at most {MAX_GRAPH_NODES} nearby nodes to keep the graph readable.")
    st.caption("Arrows point from caller to callee. Colors mark folders; dots are functions, diamonds are methods, and hexagons are unresolved calls. Dashed links are inferred by matching call names.")

    module_names = sorted({
        node["file_path"].split("/")[0] if "/" in node["file_path"] else "(root)"
        for node in visible_nodes.values() if node["symbol_type"] != "external"
    })
    module_colors = {name: PALETTE[index % len(PALETTE)] for index, name in enumerate(module_names)}
    graph_nodes = []
    for node_id, node in visible_nodes.items():
        if node["symbol_type"] == "external":
            group = "external"
            color = "#64748B"
            shape = "hexagon"
        else:
            group = node["file_path"].split("/")[0] if "/" in node["file_path"] else "(root)"
            color = module_colors[group]
            shape = "diamond" if node["symbol_type"] == "method" else "dot"
        qualified = f"{node['class_name']}.{node['symbol_name']}" if node["class_name"] else node["symbol_name"]
        title = f"{qualified}\n{node['file_path']}:{node['start_line']}-{node['end_line']}"
        if node.get("docstring"):
            title += f"\n{node['docstring'][:240]}"
        graph_nodes.append(Node(
            id=node_id,
            label=qualified,
            title=title,
            group=group,
            color=color,
            shape=shape,
            size=30 if node_id == focus_id else 20,
            font={"color": "#E5E7EB", "size": 14},
            borderWidth=2 if node_id == focus_id else 1,
        ))

    graph_edges = [Edge(
        source=edge["source"],
        target=edge["target"],
        label=edge["label"],
        color="#94A3B8" if edge["label"] == "possible call" else "#64748B",
        dashes=True,
        arrows="to",
        font={"color": "#CBD5E1", "size": 10},
        title="Inferred from a matching call name; dynamic dispatch may differ.",
    ) for edge in visible_edges]

    selected = agraph(
        nodes=graph_nodes,
        edges=graph_edges,
        config=Config(
            width=720,
            height=560,
            directed=True,
            physics=True,
            hierarchical=False,
            stabilization=True,
            interaction={"hover": True, "navigationButtons": True, "keyboard": True},
        ),
    )

    if selected in visible_nodes:
        node = visible_nodes[selected]
        if node["symbol_type"] == "external":
            st.info(f"`{node['symbol_name']}` has no matching indexed function. It may be external or dynamically resolved.")
            return

        qualified = f"{node['class_name']}.{node['symbol_name']}" if node["class_name"] else node["symbol_name"]
        st.markdown(f"#### `{qualified}`")
        st.caption(f"`{node['file_path']}:{node['start_line']}-{node['end_line']}` · {node['symbol_type']}")
        if node.get("docstring"):
            st.write(node["docstring"])
        source_path = repo_dir / Path(node["file_path"])
        if source_path.exists():
            lines = source_path.read_text(encoding="utf-8", errors="replace").splitlines()
            snippet = "\n".join(lines[max(0, node["start_line"] - 1):node["end_line"]])
            st.code(snippet, language=source_path.suffix.lstrip("."))
            github_url = f"{repo_url.rstrip('/').removesuffix('.git')}/blob/HEAD/{quote(node['file_path'])}#L{node['start_line']}-L{node['end_line']}"
            st.markdown(f"[Open this code on GitHub]({github_url})")
        ask_key = f"ask_graph_{repo_key}_{hashlib.sha1(selected.encode()).hexdigest()[:10]}"
        if st.button("Ask about this function in chat", key=ask_key):
            st.session_state["pending_graph_question"] = (
                f"Explain how `{qualified}` works in `{node['file_path']}`. "
                "Describe its callers and the functions it calls, using code evidence and citations."
            )
            st.rerun()


def _render_file_browser(files, edges, unresolved, repo_dir: Path, repo_url: str, repo_key: str):
    if not files:
        st.info("No indexed files were found.")
        return
    grouped: dict[str, list[str]] = {}
    for path in sorted(files):
        folder = str(Path(path).parent).replace("\\", "/")
        grouped.setdefault(folder, []).append(path)
    folders = sorted(grouped)
    folder = st.selectbox("Folder", folders, key=f"browse_folder_{repo_key}")
    file_path = st.selectbox("File", grouped[folder], key=f"browse_file_{repo_key}_{folder}")
    symbols = sorted(files[file_path])
    if symbols:
        st.caption("Symbols: " + " · ".join(symbols))
        for symbol in symbols:
            outgoing = [f"{target['file_path']} · {target['class_name'] + '.' if target['class_name'] else ''}{target['symbol_name']}"
                        for edge in edges if edge["source_node"]["file_path"] == file_path
                        and edge["source_node"]["symbol_name"] == symbol
                        for target in [edge["target_node"]]]
            incoming = [f"{edge['source_node']['file_path']} · {edge['source_node']['class_name'] + '.' if edge['source_node']['class_name'] else ''}{edge['source_node']['symbol_name']}"
                        for edge in edges if edge["target_node"]["file_path"] == file_path
                        and edge["target_node"]["symbol_name"] == symbol]
            external = [item["name"] for item in unresolved
                        if item["source_node"]["file_path"] == file_path
                        and item["source_node"]["symbol_name"] == symbol]
            details = []
            if outgoing:
                details.append("possible calls to " + ", ".join(sorted(set(outgoing))))
            if incoming:
                details.append("possibly called by " + ", ".join(sorted(set(incoming))))
            if external:
                details.append("unresolved calls: " + ", ".join(sorted(set(external))))
            if details:
                st.caption(f"**{symbol}** — " + "; ".join(details))
    full_path = repo_dir / Path(file_path)
    if full_path.exists():
        source = full_path.read_text(encoding="utf-8", errors="replace")
        st.code(source, language=full_path.suffix.lstrip("."))
    st.markdown(f"[Open file on GitHub]({repo_url.rstrip('/').removesuffix('.git')}/blob/HEAD/{quote(file_path)})")


def render_code_explorer(index, repo_dir: Path, repo_url: str, repo_key: str):
    files, symbols, raw_edges, raw_unresolved = build_code_map(index)
    edges = [
        {**edge, "source_node": symbols[edge["source"]], "target_node": symbols[edge["target"]]}
        for edge in raw_edges
    ]
    unresolved = [
        {**item, "source_node": symbols[item["source"]]} for item in raw_unresolved
    ]
    st.subheader("Code map")
    graph_tab, browse_tab = st.tabs(["Visual graph", "Browse files"])
    with graph_tab:
        _render_graph(symbols, raw_edges, raw_unresolved, repo_key, repo_dir, repo_url)
    with browse_tab:
        _render_file_browser(files, edges, unresolved, repo_dir, repo_url, repo_key)
