"""Builds results/report.html, a local report, from the summary and a few example items."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from eval.llm import llm_block
from eval.longbench import load

TEMPLATE = Path("eval/report_template.html")
SUMMARY = Path("results/summary.json")
ITEMS = Path("results/raw/items.jsonl")
OUT = Path("results/report.html")


def examples(rows: list[dict], frozen: dict, questions: dict[str, str]) -> list[dict]:
    """One question where Jev's selection was right and embeddings' wrong, and one the other way."""
    def pick(r, name):
        return (r["results"].get(f"{name}@{frozen[name]}") or {})

    def good(r):
        return (not r["fits"]) and r.get("strip") and 80 <= r["n_chunks"] <= 400 and pick(r, "jev") and pick(r, "embed")

    pool = sorted((r for r in rows if good(r)), key=lambda r: r["id"])
    out = []
    for title, want_jev, want_embed in (
        ("Jev's selection got it right, embeddings' did not", True, False),
        ("Embeddings' selection got it right, Jev's did not", False, True),
    ):
        for r in pool:
            jev, emb = pick(r, "jev"), pick(r, "embed")
            if (jev["choice"] == r["answer"]) == want_jev and (emb["choice"] == r["answer"]) == want_embed:
                out.append({
                    "title": title, "question": questions[r["id"]], "n_chunks": r["n_chunks"],
                    "context_tokens": r["context_tokens"], "answer": r["answer"],
                    "jev_choice": jev["choice"], "embed_choice": emb["choice"],
                    "jev": r["strip"]["jev"], "embed": r["strip"]["embed"],
                    "kept_jev": jev["kept"], "kept_embed": emb["kept"],
                })
                break
    return out


def build(spend_usd: float) -> None:
    summary = json.loads(SUMMARY.read_text())
    rows = [json.loads(l) for l in ITEMS.read_text().splitlines() if l.strip()]
    questions = {i["_id"]: i["question"].strip() for i in load()}
    llm = llm_block(summary)
    data = {
        **summary, "llm": llm, "model": "typesafe/jev-1.13", "spend_usd": spend_usd, "date": date.today().isoformat(),
        "examples": examples(rows, summary["frozen_budgets"], questions),
        "footer": f"longjev v0.1 · LongBench v2 (zai-org, Apache-2.0) · {summary['total_chunks']:,} chunks scored, {summary['failed_chunks']} failed",
    }
    html = TEMPLATE.read_text().replace("/*DATA*/null", json.dumps(data, ensure_ascii=False).replace("</", "<\\/"))
    OUT.write_text(html)
    print(f"wrote {OUT} ({len(html)/1024:.0f} KB, {len(data['examples'])} examples)")


if __name__ == "__main__":
    import sys
    build(float(sys.argv[1]) if len(sys.argv) > 1 else 0.0)
