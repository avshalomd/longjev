"""Published LLM scores on LongBench v2 with cost per question estimated from list prices."""

from __future__ import annotations

import json
from pathlib import Path

REFERENCE = Path("eval/llm_reference.json")


def llm_block(summary: dict) -> dict:
    ref = json.loads(REFERENCE.read_text())
    short = summary["categories"]["short"]
    tokens = short["context_tokens_capped_mean"]
    out_tokens = ref["assumed_output_tokens"]
    return {
        "human_short": ref["human"]["short"], "tokens_mean": tokens, "n": short["n"], "n_over_cap": short["n_over_cap"],
        "source": ref["source"], "prices": ref["prices"], "assumed_output_tokens": out_tokens,
        "longjev": {
            k: {"acc": v["acc"], "k": v["k"], "n": v["n"], "low": v["low"], "high": v["high"], "usd": v["usd_mean"]}
            for k, v in short["selectors"].items()
        },
        "newer_source": ref.get("newer_source"),
        "newer": [
            {**m, "usd": (tokens * m["in"] + out_tokens["reasoning"] * m["out"]) / 1e6} for m in ref.get("newer", [])
        ],
        "models": [
            {**m, "usd": (tokens * m["in"] + out_tokens[m["mode"]] * m["out"]) / 1e6} for m in ref["models"]
        ],
    }
