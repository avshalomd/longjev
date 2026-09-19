"""Turns results/raw/items.jsonl into results/summary.json (tables, intervals, tests)."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import median

SELECTORS = ("truncate", "embed", "jev", "embed_then_jev")
BUDGETS = (8_000, 16_000, 28_000)
JEV_USD, EMBED_USD = 0.042 / 1e6, 0.01 / 1e6
ITEMS = Path("results/raw/items.jsonl")
OUT = Path("results/summary.json")


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


def mcnemar(b: int, c: int) -> float:
    """Exact two-sided p-value from the discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(0, min(b, c) + 1)) / 2**n
    return min(1.0, 2 * tail)


def acc(rows: list[dict], pick) -> dict:
    hits = [pick(r) == r["answer"] for r in rows if pick(r) is not None]
    k, n = sum(hits), len(hits)
    low, high = wilson(k, n)
    return {"k": k, "n": n, "acc": k / n if n else None, "low": low, "high": high}


def choice_at(name: str, budget: int):
    return lambda r: (r["results"].get(f"{name}@{budget}") or {}).get("choice")


def widened(name: str, budget: int):
    def pick(r):
        first = r["results"].get(f"{name}@{budget}")
        if not first:
            return None
        if first["suff"] < 0.5 and budget < BUDGETS[-1]:
            return (r["results"].get(f"{name}@{BUDGETS[-1]}") or first)["choice"]
        return first["choice"]
    return pick


def item_cost(r: dict, name: str, budget: int) -> float | None:
    final = r["results"].get(f"{name}@{budget}")
    if not final:
        return None
    cost = (final["state_tokens"] + 600) * JEV_USD
    scoring = (r.get("jev_scoring_tokens") or r.get("timing", {}).get("jev", {}).get("tokens") or 0) * JEV_USD
    if name in ("embed", "embed_then_jev"):
        cost += r["context_tokens"] * EMBED_USD
    if name == "jev":
        cost += scoring
    if name == "embed_then_jev":
        cost += scoring * r["shortlist"] / max(1, r["n_chunks"])
    return cost


def build() -> dict:
    rows = [json.loads(l) for l in ITEMS.read_text().splitlines() if l.strip()]
    long_rows = [r for r in rows if not r["fits"]]
    dev = [r for r in long_rows if r["split"] == "dev"]
    test = [r for r in long_rows if r["split"] == "test"]

    sweep = {
        split: {s: {str(b): acc(group, choice_at(s, b)) for b in BUDGETS} for s in SELECTORS}
        for split, group in (("dev", dev), ("test", test), ("all", long_rows))
    }
    frozen = {}
    for s in SELECTORS:
        best = max(BUDGETS, key=lambda b: ((sweep["dev"][s][str(b)]["acc"] or 0), -b))
        frozen[s] = best

    def breakdown(group, pick):
        out = {}
        for field in ("length", "difficulty", "domain"):
            buckets = defaultdict(list)
            for r in group:
                buckets[r[field]].append(r)
            out[field] = {k: acc(v, pick) for k, v in sorted(buckets.items())}
        return out

    headline = {}
    for s in SELECTORS:
        pick = choice_at(s, frozen[s])
        headline[s] = {
            "budget": frozen[s], "test": acc(test, pick), "all": acc(long_rows, pick),
            "dev_widened": acc(dev, widened(s, frozen[s])), "dev": acc(dev, pick),
            "by": breakdown(long_rows, pick),
        }
    vote_pick = lambda r: (r.get("vote") or {}).get("choice")
    vote_short = lambda r: (r.get("vote_shortlist") or {}).get("choice")
    headline["vote"] = {"test": acc(test, vote_pick), "all": acc(long_rows, vote_pick), "by": breakdown(long_rows, vote_pick)}
    headline["vote_shortlist"] = {"test": acc(test, vote_short), "all": acc(long_rows, vote_short)}

    def paired(group, a, b):
        both = [(a(r) == r["answer"], b(r) == r["answer"]) for r in group if a(r) and b(r)]
        only_a = sum(1 for x, y in both if x and not y)
        only_b = sum(1 for x, y in both if y and not x)
        return {"n": len(both), "only_first": only_a, "only_second": only_b, "p": mcnemar(only_a, only_b)}

    pairs = {}
    for group_name, group in (("test", test), ("all", long_rows)):
        pairs[group_name] = {
            "jev_vs_embed": paired(group, choice_at("jev", frozen["jev"]), choice_at("embed", frozen["embed"])),
            "jev_vs_truncate": paired(group, choice_at("jev", frozen["jev"]), choice_at("truncate", frozen["truncate"])),
            "embed_vs_truncate": paired(group, choice_at("embed", frozen["embed"]), choice_at("truncate", frozen["truncate"])),
            "embed_then_jev_vs_embed": paired(group, choice_at("embed_then_jev", frozen["embed_then_jev"]), choice_at("embed", frozen["embed"])),
            "jev_vs_vote": paired(group, choice_at("jev", frozen["jev"]), vote_pick),
        }

    jev_pick = choice_at("jev", frozen["jev"])
    agree = [r for r in long_rows if jev_pick(r) and vote_pick(r) and jev_pick(r) == vote_pick(r)]
    differ = [r for r in long_rows if jev_pick(r) and vote_pick(r) and jev_pick(r) != vote_pick(r)]
    agreement = {
        "agree": acc(agree, jev_pick), "disagree_final": acc(differ, jev_pick), "disagree_vote": acc(differ, vote_pick),
    }

    control_rows = [r for r in rows if r["fits"] and r.get("direct")]
    control = {"direct": acc(control_rows, lambda r: r["direct"]["choice"])}
    for b in BUDGETS[:2]:
        sub = [r for r in control_rows if r["results"].get(f"jev@{b}")]
        if not sub:  # the control ran at the smallest budget only
            continue
        control[str(b)] = {
            "direct_same_items": acc(sub, lambda r: r["direct"]["choice"]),
            **{s: acc(sub, choice_at(s, b)) for s in SELECTORS},
        }

    def med(values):
        values = [v for v in values if v is not None]
        return median(values) if values else None

    cost = {}
    for s in SELECTORS:
        b = frozen[s]
        cost[s] = {
            "usd_median": med(item_cost(r, s, b) for r in long_rows),
            "usd_mean": (lambda v: sum(v) / len(v) if v else None)([c for c in (item_cost(r, s, b) for r in long_rows) if c is not None]),
            "final_seconds_median": med((r["results"].get(f"{s}@{b}") or {}).get("seconds") for r in long_rows),
        }
    live = [r for r in long_rows if (r["timing"]["jev"].get("requests") or 0) > 0.5 * r["n_chunks"]]
    timing = {
        "n_live": len(live),
        "jev_scoring_seconds_median": med(r["timing"]["jev"]["seconds"] for r in live),
        "chunks_median": med(r["n_chunks"] for r in long_rows),
        "shortlist_median": med(r["shortlist"] for r in long_rows),
        "scoring_tokens_median": med(r.get("jev_scoring_tokens") for r in long_rows),
        "context_tokens_median": med(r["context_tokens"] for r in long_rows),
        "embed_seconds_median": med(r["timing"]["embed"]["seconds"] for r in long_rows if r["timing"]["embed"]["requests"]),
    }

    # how accuracy moves with context size, per selector
    bins = [(0, 50_000), (50_000, 90_000), (90_000, 10**9)]
    by_size = []
    for low, high in bins:
        group = [r for r in long_rows if low <= r["context_tokens"] < high]
        if group:
            by_size.append({
                "range": [low, high], "n": len(group),
                **{s: acc(group, choice_at(s, frozen[s])) for s in SELECTORS},
                "vote": acc(group, vote_pick),
            })

    shares = [sum(v > 0.5 for v in r["strip"]["jev"]) / len(r["strip"]["jev"]) for r in long_rows if r.get("strip")]
    overlaps = []
    for r in long_rows:
        a, b = r["results"].get("jev@16000"), r["results"].get("embed@16000")
        if a and b:
            overlaps.append(len(set(a["kept"]) & set(b["kept"])) / max(1, len(set(a["kept"]) | set(b["kept"]))))
    not_better = sum(
        1 for s in SELECTORS
        if (sweep["dev"][s]["28000"]["acc"] or 0) <= (sweep["dev"][s]["16000"]["acc"] or 0)
    )
    extras = {
        "relevant_share_median": med(shares), "overlap_median": med(overlaps), "overlap_n": len(overlaps),
        "selectors_not_better_at_28k": not_better,
    }

    # whole length categories, the way a user meets them: inputs that fit go straight to Jev
    def served(name):
        def pick(r):
            if r["fits"]:
                return (r.get("direct") or {}).get("choice")
            return choice_at(name, frozen[name])(r)
        return pick

    def served_cost(r, name):
        if r["fits"]:
            return (r["context_tokens"] + 600) * JEV_USD
        return item_cost(r, name, frozen[name])

    categories = {}
    for length in ("short", "medium", "long"):
        group = [r for r in rows if r["length"] == length]
        if not group:
            continue
        categories[length] = {
            "n": len(group), "n_fit": sum(r["fits"] for r in group),
            "context_tokens_mean": sum(r["context_tokens"] for r in group) / len(group),
            # LongBench truncates an input to the model's window; 128K is the common one
            "context_tokens_capped_mean": sum(min(r["context_tokens"], 128_000) for r in group) / len(group),
            "n_over_cap": sum(r["context_tokens"] > 128_000 for r in group),
            "selectors": {
                s: {
                    **acc(group, served(s)),
                    "usd_mean": sum(served_cost(r, s) or 0 for r in group) / len(group),
                }
                for s in SELECTORS
            },
        }

    latency = None
    lat_path = Path("results/raw/latency.jsonl")
    if lat_path.exists():
        lat = [json.loads(l) for l in lat_path.read_text().splitlines() if l.strip()]
        reduced = [x for x in lat if not x["passthrough"]]
        latency = {
            "n_questions": len({x["id"] for x in reduced}),
            "passthrough_seconds_median": med(x["seconds"] for x in lat if x["passthrough"]),
            "n_passthrough": sum(1 for x in lat if x["passthrough"]),
            "selectors": {
                s: {
                    "seconds_median": med(x["seconds"] for x in reduced if x["selector"] == s),
                    "seconds_max": max((x["seconds"] for x in reduced if x["selector"] == s), default=None),
                    "requests_median": med(x["requests"] for x in reduced if x["selector"] == s),
                    "usd_mean": (lambda v: sum(v) / len(v) if v else None)([x["usd"] for x in reduced if x["selector"] == s]),
                    "widened": sum(1 for x in reduced if x["selector"] == s and x["widened"]),
                    "n": sum(1 for x in reduced if x["selector"] == s),
                }
                for s in SELECTORS
            },
            "points": [
                {"s": x["selector"], "tokens": x["context_tokens"], "seconds": x["seconds"], "requests": x["requests"]}
                for x in reduced
            ],
        }

    idx_path = Path("results/raw/latency_indexed.jsonl")
    if latency and idx_path.exists():
        # same questions, chunk vectors read from a prebuilt index; only the query is embedded live
        idx = [json.loads(l) for l in idx_path.read_text().splitlines() if l.strip()]
        for s in SELECTORS:
            seconds = [x["seconds"] for x in idx if x["selector"] == s and not x["passthrough"]]
            if seconds:
                latency["selectors"][s]["indexed"] = {
                    "seconds_median": med(seconds), "seconds_max": max(seconds), "n": len(seconds),
                    "requests_median": med(x["requests"] for x in idx if x["selector"] == s and not x["passthrough"]),
                }

    summary = {
        "extras": extras, "categories": categories, "latency": latency,
        "n_items": len(rows), "n_long": len(long_rows), "n_dev": len(dev), "n_test": len(test),
        "n_control": len(control_rows), "max_context_tokens": max(r["context_tokens"] for r in rows),
        "failed_chunks": sum(r.get("failed_chunks", 0) for r in rows),
        "total_chunks": sum(r["n_chunks"] for r in rows),
        "frozen_budgets": frozen, "sweep": sweep, "headline": headline, "pairs": pairs,
        "agreement": agreement, "control": control, "cost": cost, "timing": timing, "by_size": by_size,
        "mix": {f: dict(sorted(_count(long_rows, f).items())) for f in ("length", "difficulty", "domain")},
    }
    OUT.write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: summary[k] for k in ("n_items", "n_long", "n_dev", "n_test", "n_control", "frozen_budgets")}, indent=1))
    for s in SELECTORS:
        h = headline[s]["test"]
        print(f"{s:16s} @{frozen[s]:>6}  test {h['k']}/{h['n']} = {100*(h['acc'] or 0):.1f}%  [{100*h['low']:.0f}–{100*h['high']:.0f}]")
    h = headline["vote"]["test"]
    print(f"{'vote':16s}          test {h['k']}/{h['n']} = {100*(h['acc'] or 0):.1f}%")
    return summary


def _count(rows, field):
    out = defaultdict(int)
    for r in rows:
        out[r[field]] += 1
    return out


if __name__ == "__main__":
    build()
