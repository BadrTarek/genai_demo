"""Pure helpers for the Streamlit app: graph rendering, charts and labels.

No Streamlit calls here, so everything is unit-testable.
"""

from typing import Any

import altair as alt
import pandas as pd

# Reference palette (validated): one hue for magnitude, neutrals for de-emphasis,
# a reserved status red for failures.
BLUE = "#2a78d6"
BLUE_FILL = "#cde2fb"
MUTED = "#9a9893"
NEUTRAL_FILL = "#f4f4f2"
NEUTRAL_INK = "#52514e"
CRITICAL = "#e34948"
CRITICAL_FILL = "#fbdada"

STATUS_BADGES: dict[str, tuple[str, str, str]] = {
    # status: (label, icon, badge color)
    "ok": ("Answered", ":material/check_circle:", "green"),
    "blocked": ("Blocked by guardrail", ":material/block:", "red"),
    "no_context": ("Not in knowledge base", ":material/search_off:", "gray"),
    "ungrounded": ("Not grounded, fallback shown", ":material/warning:", "orange"),
    "error": ("Error", ":material/error:", "red"),
}

NODE_LABELS = {
    "input_guardrails": "Input guardrails",
    "retrieve": "Retrieve",
    "generate": "Generate",
    "output_guardrails": "Output guardrails",
    "evaluate": "Online evaluation",
    "finalize": "Finalize",
}


def status_badge(status: str | None) -> tuple[str, str, str]:
    return STATUS_BADGES.get(status or "", (status or "unknown", ":material/help:", "gray"))


# -- agent graph -------------------------------------------------------------------


def graph_dot(graph: Any, path: list[str] | None = None, failed_node: str | None = None) -> str:
    """Graphviz DOT for a LangGraph `get_graph()` result.

    `path` is the ordered list of executed nodes; their nodes and edges are highlighted.
    `failed_node` is drawn in the status red (where a guardrail stopped the run).
    """
    path = path or []
    walked = ["__start__", *path, "__end__"] if path else []
    visited_edges = set(zip(walked, walked[1:], strict=False))
    visits = {node: path.count(node) for node in set(path)}

    lines = [
        "digraph G {",
        '  graph [rankdir=TB, bgcolor="transparent", nodesep=0.35, ranksep=0.35];',
        '  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11,'
        f' penwidth=1.2, color="{MUTED}", fillcolor="{NEUTRAL_FILL}", fontcolor="{NEUTRAL_INK}"];',
        f'  edge [color="{MUTED}", arrowsize=0.6, penwidth=1];',
    ]
    for node_id in graph.nodes:
        if node_id in ("__start__", "__end__"):
            label = "start" if node_id == "__start__" else "end"
            attrs = [f'label="{label}"', "shape=circle", "width=0.45", "fixedsize=true"]
            if walked:
                attrs.append(f'color="{BLUE}"')
        else:
            label = NODE_LABELS.get(node_id, node_id)
            if visits.get(node_id, 0) > 1:
                label += f" ×{visits[node_id]}"
            attrs = [f'label="{label}"']
            if node_id == failed_node:
                attrs += [f'color="{CRITICAL}"', f'fillcolor="{CRITICAL_FILL}"', "penwidth=2"]
                attrs.append('fontcolor="#0b0b0b"')
            elif node_id in visits:
                attrs += [f'color="{BLUE}"', f'fillcolor="{BLUE_FILL}"', "penwidth=2"]
                attrs.append('fontcolor="#0b0b0b"')
        lines.append(f'  "{node_id}" [{", ".join(attrs)}];')

    for edge in graph.edges:
        attrs = []
        if edge.conditional:
            attrs.append("style=dashed")
        if (edge.source, edge.target) in visited_edges:
            attrs += [f'color="{BLUE}"', "penwidth=2.2", "style=solid"]
        lines.append(f'  "{edge.source}" -> "{edge.target}" [{", ".join(attrs)}];')
    lines.append("}")
    return "\n".join(lines)


def failed_node_for(status: str | None) -> str | None:
    return {"blocked": "input_guardrails", "ungrounded": "output_guardrails"}.get(status or "")


def describe_update(node: str, update: dict[str, Any]) -> str:
    """One-line, human-readable summary of what a node did."""
    status = update.get("status")
    if node == "input_guardrails":
        if status == "blocked":
            return update.get("answer", "Blocked")
        redactions = [e["reason"] for e in update.get("guardrail_events", []) if e.get("modified")]
        return "Passed" + (f" · {'; '.join(redactions)}" if redactions else "")
    if node == "retrieve":
        if status == "no_context":
            return "No passage above the relevance threshold"
        if status == "error":
            return "Retrieval failed"
        return f"{len(update.get('contexts', []))} relevant passages"
    if node == "generate":
        return f"Draft #{update.get('attempts', 1)} · {len(update.get('answer', ''))} characters"
    if node == "output_guardrails":
        score = update.get("evaluations", {}).get("groundedness", {}).get("score")
        grounded = f" · groundedness {score:.2f}" if score is not None else ""
        if status == "ok":
            return f"All checks passed{grounded}"
        if status == "retry":
            return f"Rejected, regenerating: {update.get('feedback', '')}"
        if status == "no_context":
            return "Model reported the answer is not in the sources"
        return f"Rejected again, safe fallback used{grounded}"
    if node == "evaluate":
        scores = update.get("evaluations", {})
        return " · ".join(f"{name} {e['score']:.2f}" for name, e in scores.items())
    if node == "finalize":
        return "Reply ready"
    return ""


# -- charts ------------------------------------------------------------------------


def _source_label(index: int, context: dict[str, Any]) -> str:
    section = context.get("section")
    source = context.get("source", "?")
    return f"[{index}] {source}" + (f" > {section}" if section else "")


def passages_frame(contexts: list[dict[str, Any]], threshold: float) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "passage": _source_label(i, c),
                "score": float(c.get("score", 0.0)),
                "used": float(c.get("score", 0.0)) >= threshold,
                "content": c.get("content", ""),
            }
            for i, c in enumerate(contexts, start=1)
        ]
    )


def score_bars(
    frame: pd.DataFrame,
    label: str,
    value: str,
    title: str,
    threshold: float | None = None,
    highlight: str | None = None,
    domain: tuple[float, float] = (0.0, 1.0),
) -> alt.Chart:
    """Horizontal one-hue bars with value labels and an optional dashed threshold rule.

    `highlight` names a boolean column; rows where it is False are drawn muted.
    """
    color = (
        alt.condition(f"datum.{highlight}", alt.value(BLUE), alt.value(MUTED))
        if highlight
        else alt.value(BLUE)
    )
    base = alt.Chart(frame).encode(
        y=alt.Y(f"{label}:N", sort=None, title=None, axis=alt.Axis(labelLimit=320)),
        x=alt.X(
            f"{value}:Q",
            title=title,
            scale=alt.Scale(domain=list(domain)),
            axis=alt.Axis(grid=True, gridOpacity=0.25, tickCount=5),
        ),
        tooltip=[alt.Tooltip(f"{label}:N"), alt.Tooltip(f"{value}:Q", format=".2f")],
    )
    bars = base.mark_bar(cornerRadiusEnd=4, size=16).encode(color=color)
    text = base.mark_text(align="left", dx=4, fontSize=11).encode(
        text=alt.Text(f"{value}:Q", format=".2f")
    )
    layers = [bars, text]
    if threshold is not None:
        rule = (
            alt.Chart(pd.DataFrame({"threshold": [threshold]}))
            .mark_rule(strokeDash=[4, 4], color=NEUTRAL_INK, size=1.5)
            .encode(
                x="threshold:Q",
                tooltip=[alt.Tooltip("threshold:Q", title="Threshold", format=".2f")],
            )
        )
        layers.append(rule)
    return alt.layer(*layers).properties(height=bar_chart_height(len(frame)) - 60)


def bar_chart_height(rows: int) -> int:
    """Pixel height for `score_bars` (30px per bar plus the x axis). Pass it to
    `st.altair_chart(height=...)`: Streamlit otherwise squashes charts, especially
    ones first rendered inside a hidden tab."""
    return 30 * max(rows, 1) + 60


def metrics_frame(summary: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"metric": name, "mean": m["mean"], "pass_rate": m["pass_rate"], "n": m["n"]}
            for name, m in summary.get("metrics", {}).items()
        ]
    )


def cases_frame(report: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for case in report.get("cases", []):
        row: dict[str, Any] = {
            "case": case["id"],
            "status": case["status"],
            "attempts": case["attempts"],
            "latency_s": case["latency_s"],
        }
        for name, score in case["scores"].items():
            row[name] = None if score["skipped"] else score["score"]
        row["failures"] = sum(
            1 for s in case["scores"].values() if not s["passed"] and not s["skipped"]
        )
        rows.append(row)
    return pd.DataFrame(rows)
