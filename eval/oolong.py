"""Spike: Oolong-synth (trec_coarse) as a map-then-aggregate workflow.

Oolong (arXiv 2511.02817) puts thousands of short records in one context and asks aggregate questions
about them: counts per label, most common label, per-user breakdowns. One-prompt LLMs score under 50%
at 131K tokens. Here code parses the records, a model labels them, and code answers the questions.

Arms share the parser and the aggregation code; only the labeller differs:
  jev            Jev labels every record with a Choice question, `batch` records per request
  jev_single     the same, one record per request
  llm:<model>    a cheap LLM labels the records in batches, as JSON
  oracle         gold labels, to check the aggregation code against the official answers

  uv run --with duckdb python -m eval.oolong fetch          # about 6 MB from Hugging Face
  uv run python -m eval.oolong tune                         # batch size, on the 8K and 16K windows
  uv run python -m eval.oolong run --batch 50 --llm-batch 50   # every window up to 262K tokens
  uv run python -m eval.oolong report
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import json
import re
import time
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

from longjev import Cache, Choice
from longjev.transport import CachedTransport, OpenRouterTransport, make_transport

MODEL = "typesafe/jev-1.13"
JEV_USD = 0.042e-6
LLMS = {  # OpenRouter list prices per token, read 2026-09-18
    "qwen/qwen3-235b-a22b-2507": (0.087e-6, 0.35e-6),
    "google/gemini-2.5-flash": (0.30e-6, 2.50e-6),
}
DATA = Path("data/oolong_trec_coarse.json")
OUT = Path("results/raw/oolong.jsonl")
TUNE = Path("results/raw/oolong_tune.jsonl")
SUMMARY = Path("results/oolong_summary.json")
TUNE_LENGTHS = (8192, 16384)
RECORD = re.compile(r"^Date: (.+?) \|\| User: (\d+) \|\| Instance: (.*?)(?: \|\| Label: (.+))?$")


# ---- the "chunker": records, not token windows ----

def parse(text: str) -> tuple[list[str], list[dict]]:
    """Label set (from the header) and one dict per record line."""
    labels = re.findall(r"'([^']+)'", text.split("\n\n", 1)[0])
    records = []
    for line in text.split("\n"):
        m = RECORD.match(line)
        if m:
            records.append({"date": m.group(1), "user": m.group(2), "text": m.group(3), "gold": m.group(4)})
    return labels, records


def load() -> tuple[list[dict], list[dict]]:
    data = json.loads(DATA.read_text())
    windows = []
    for c in data["contexts"]:
        labels, records = parse(c["text_with_labels"])
        windows.append({"context_len": c["context_len"], "window": c["context_window_id"], "labels": labels,
                        "records": records, "chars": len(c["text"])})
    return windows, data["questions"]


# ---- labellers ----

async def jev_label(transport, records: list[dict], labels: list[str], batch: int, concurrency: int) -> dict:
    gate = asyncio.Semaphore(concurrency)
    predicted: list[str | None] = [None] * len(records)
    probabilities: list[dict | None] = [None] * len(records)
    tokens = 0

    async def one(start: int):
        nonlocal tokens
        group = records[start:start + batch]
        if batch == 1:
            state = group[0]["text"]
            questions = {"q1": Choice("The kind of answer this question is asking for.", labels).to_json()}
        else:
            state = "\n".join(f"{i + 1}. {r['text']}" for i, r in enumerate(group))
            questions = {f"q{i + 1}": Choice(f"The kind of answer that question {i + 1} is asking for.", labels).to_json()
                         for i in range(len(group))}
        async with gate:
            data = await transport.decide(state, questions, MODEL)
        tokens += int((data.get("usage") or {}).get("input_tokens", 0) or 0)
        for i in range(len(group)):
            answer = data["answers"].get(f"q{i + 1}") or {}
            predicted[start + i], probabilities[start + i] = answer.get("choice"), answer.get("probabilities")

    hits, started = transport.usage.cache_hits, time.perf_counter()
    await asyncio.gather(*(one(s) for s in range(0, len(records), batch)))
    return {"predicted": predicted, "probabilities": probabilities, "usd": tokens * JEV_USD, "tokens": tokens, "seconds": round(time.perf_counter() - started, 2),
            "requests": -(-len(records) // batch), "live": transport.usage.cache_hits == hits}


async def llm_label(llm: OpenRouterTransport, model: str, records: list[dict], labels: list[str], batch: int, concurrency: int) -> dict:
    letters = "ABCDEFGHIJ"[:len(labels)]
    legend = "\n".join(f"{letter} = {label}" for letter, label in zip(letters, labels))
    gate = asyncio.Semaphore(concurrency)
    predicted: list[str | None] = [None] * len(records)
    totals = {"usd": 0.0, "prompt_tokens": 0, "completion_tokens": 0}

    async def one(start: int):
        group = records[start:start + batch]
        listing = "\n".join(f"{i + 1}. {r['text']}" for i, r in enumerate(group))
        body = {
            "model": model, "temperature": 0, "max_tokens": 12 * len(group) + 50, "usage": {"include": True},
            "messages": [
                {"role": "system", "content": "You label general-knowledge questions by the kind of answer they ask for. Answer only with a JSON object."},
                {"role": "user", "content": (
                    f"Categories:\n{legend}\n\nQuestions:\n{listing}\n\n"
                    "Reply with one JSON object mapping each question number to its category letter. No other text."
                )},
            ],
        }
        if "gemini" in model:
            body["reasoning"] = {"enabled": False}
        async with gate:
            data = await llm._post("/v1/chat/completions", body)
        text = data["choices"][0]["message"].get("content") or ""
        try:
            parsed = json.loads(text[text.index("{"): text.rindex("}") + 1])
        except ValueError:
            parsed = {}
        for i in range(len(group)):
            letter = str(parsed.get(str(i + 1), "")).strip().upper()[:1]
            predicted[start + i] = labels[letters.index(letter)] if letter and letter in letters else None
        usage = data.get("usage") or {}
        cost = usage.get("cost")
        if cost is None:
            cost = usage.get("prompt_tokens", 0) * LLMS[model][0] + usage.get("completion_tokens", 0) * LLMS[model][1]
        totals["usd"] += float(cost)
        totals["prompt_tokens"] += usage.get("prompt_tokens", 0)
        totals["completion_tokens"] += usage.get("completion_tokens", 0)

    started = time.perf_counter()
    await asyncio.gather(*(one(s) for s in range(0, len(records), batch)))
    return {"predicted": predicted, **totals, "tokens": totals["prompt_tokens"] + totals["completion_tokens"],
            "seconds": round(time.perf_counter() - started, 2), "requests": -(-len(records) // batch), "live": True}


# ---- aggregation: plain code over (record, label) pairs ----

def solve(question: str, records: list[dict], predicted: list[str | None], probabilities: list[dict | None] | None = None) -> str:
    """With `probabilities`, a record adds its probability to each label's count instead of one vote."""
    weights = [w or {p: 1.0} for w, p in zip(probabilities, predicted)] if probabilities else [{p: 1.0} for p in predicted]
    rows = list(zip(records, weights))
    subset = re.search(r"user IDs ((?:\d+(?:, and |, | and )?)+)", question)
    if subset:
        users = set(re.findall(r"\d+", subset.group(1)))
        rows = [(r, w) for r, w in rows if r["user"] in users]
    counts: Counter = Counter()
    for _, w in rows:
        counts.update(w)

    m = re.search(r"which user has the most instances with the label (.+?)\?", question)
    if m:
        per_user: Counter = Counter()
        for r, w in rows:
            per_user[r["user"]] += w.get(m.group(1), 0.0)
        return f"User: {per_user.most_common(1)[0][0]}"
    if "which user is represented" in question:
        ranked = Counter(r["user"] for r, _ in rows).most_common()
        return f"User: {ranked[1 if 'second most' in question else 0][0]}"
    m = re.search(r"is label '(.+?)' more common, less common, or the same frequency as label '(.+?)'\?", question)
    if m:
        a, b = counts[m.group(1)], counts[m.group(2)]
        return "Answer: " + ("more common than" if a > b else "less common than" if a < b else "same frequency as")
    m = re.search(r"how many data points should be classified as label '(.+?)'", question)
    if m:
        return f"Answer: {round(counts[m.group(1)])}"
    m = re.search(r"which of the labels is the (most|least) common\?.*one of the labels: (.+?)\.?\s*$", question, re.S)
    if m:
        options = [o.strip() for o in m.group(2).split(", ")]
        pick = (max if m.group(1) == "most" else min)(options, key=lambda o: counts[o])
        return f"Label: {pick}"
    raise ValueError(f"no rule for: {question[:120]}")


def official_score(question: dict, output: str) -> float:
    """Mirrors synth_process_response in the Oolong repo: exact match against the first gold answer,
    and 0.75 ** |error| for numeric answers."""
    gold = ast.literal_eval(question["answer"])[0]
    answer = output.split(":")[-1].strip()
    if str(answer) == str(gold):
        return 1.0
    if question["answer_type"] == "ANSWER_TYPE.NUMERIC":
        try:
            return 0.75 ** abs(int(gold) - int(answer))
        except ValueError:
            return 0.0
    return 0.0


def grade(window: dict, questions: list[dict], predicted: list[str | None], probabilities=None) -> dict:
    mine = [q for q in questions if q["context_len"] == window["context_len"] and q["context_window_id"] == window["window"]]
    scores = {str(q["id"]): official_score(q, solve(q["question"], window["records"], predicted)) for q in mine}
    gold = [r["gold"] for r in window["records"]]
    soft = {"scores_soft": {str(q["id"]): official_score(q, solve(q["question"], window["records"], predicted, probabilities)) for q in mine}} if probabilities else {}
    return {**soft, "scores": scores, "record_accuracy": sum(p == g for p, g in zip(predicted, gold)) / len(gold),
            "unlabelled": sum(p is None for p in predicted)}


# ---- commands ----

def cmd_fetch(_args) -> None:
    """The trec_coarse rows of oolongbench/oolong-synth (validation split), contexts up to 262K tokens.
    Reads only the needed rows of the remote parquet files; the full dataset is about 12 GB."""
    import duckdb

    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")
    src = "read_parquet('hf://datasets/oolongbench/oolong-synth/data/validation-*.parquet')"
    where = "dataset = 'trec_coarse' and context_len <= 262144"
    cols = ["id", "context_len", "context_window_id", "question", "task_group", "task", "answer", "answer_type", "input_subset", "num_labels"]
    questions = [dict(zip(cols, r)) for r in con.execute(f"select {', '.join(cols)} from {src} where {where} order by context_len, id").fetchall()]
    contexts = [{"context_len": r[0], "context_window_id": r[1], "text": r[2], "text_with_labels": r[3]} for r in con.execute(
        f"select context_len, context_window_id, any_value(context_window_text), any_value(context_window_text_with_labels) from {src} where {where} group by all order by 1, 2").fetchall()]
    DATA.parent.mkdir(exist_ok=True)
    DATA.write_text(json.dumps({"questions": questions, "contexts": contexts}))
    print(f"{len(questions)} questions, {len(contexts)} contexts -> {DATA}")


def transports():
    load_dotenv(".env")
    # Jev goes to the provider chosen by LONGJEV_PROVIDER; the LLM baselines stay on OpenRouter.
    llm = OpenRouterTransport()
    return CachedTransport(make_transport(), Cache(".longjev_cache/bench.sqlite")), llm


async def cmd_tune(args) -> None:
    windows, questions = load()
    jev, llm = transports()
    rows = []
    for window in [w for w in windows if w["context_len"] in TUNE_LENGTHS]:
        for batch in args.batches:
            result = await jev_label(jev, window["records"], window["labels"], batch, args.concurrency)
            rows.append({"arm": "jev", "batch": batch, **_row(window, questions, result)})
        for model in LLMS:
            for batch in args.llm_batches:
                result = await llm_label(llm, model, window["records"], window["labels"], batch, args.concurrency)
                rows.append({"arm": f"llm:{model}", "batch": batch, **_row(window, questions, result)})
        print(f"  tuned on {window['context_len']} / window {window['window']}", flush=True)
    TUNE.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"{'arm':<32}{'batch':>6}{'records right':>15}{'task score':>12}{'soft counts':>13}{'¢ per 1K records':>18}{'sec per 1K':>12}")
    for arm, batch in sorted({(r["arm"], r["batch"]) for r in rows}):
        group = [r for r in rows if (r["arm"], r["batch"]) == (arm, batch)]
        n = sum(r["n_records"] for r in group)
        scores = [s for r in group for s in r["scores"].values()]
        soft = [s for r in group for s in r.get("scores_soft", {}).values()]
        print(f"{arm:<32}{batch:>6}{100 * sum(r['record_accuracy'] * r['n_records'] for r in group) / n:>14.1f}%"
              f"{100 * sum(scores) / len(scores):>11.1f}%{(f'{100 * sum(soft) / len(soft):.1f}%' if soft else '-'):>13}{100 * sum(r['usd'] for r in group) / n * 1000:>17.3f}¢"
              f"{sum(r['seconds'] for r in group) / n * 1000:>12.1f}")
    await jev.aclose(); await llm.aclose()


def _row(window: dict, questions: list[dict], result: dict) -> dict:
    predicted, probabilities = result.pop("predicted"), result.pop("probabilities", None)
    return {"context_len": window["context_len"], "window": window["window"], "n_records": len(window["records"]),
            **result, **grade(window, questions, predicted, probabilities), "label_counts": dict(Counter(str(p) for p in predicted))}


async def cmd_run(args) -> None:
    windows, questions = load()
    jev, llm = transports()
    done = {(r["arm"], r["context_len"], r["window"]) for r in map(json.loads, OUT.read_text().splitlines())} if OUT.exists() else set()
    spent = 0.0
    with OUT.open("a") as handle:
        for window in windows:
            arms = {"oracle": None, "jev": args.batch, "jev_single": 1, **{f"llm:{m}": args.llm_batch for m in LLMS}}
            for arm, batch in arms.items():
                if (arm, window["context_len"], window["window"]) in done:
                    continue
                if window["context_len"] > args.costly_max_len and (arm == "jev_single" or "gemini" in arm):
                    continue  # the two dearest labellers stop at 131K
                if arm == "oracle":
                    result = {"predicted": [r["gold"] for r in window["records"]], "usd": 0.0, "tokens": 0, "seconds": 0.0, "requests": 0, "live": False}
                elif arm.startswith("jev"):
                    result = await jev_label(jev, window["records"], window["labels"], batch, args.concurrency)
                else:
                    result = await llm_label(llm, arm[4:], window["records"], window["labels"], batch, args.concurrency)
                row = {"arm": arm, "batch": batch, **_row(window, questions, result)}
                spent += row["usd"]
                handle.write(json.dumps(row) + "\n"); handle.flush()
                print(f"  {window['context_len']:>7} w{window['window']:<3}{arm:<30} records {100 * row['record_accuracy']:5.1f}%  "
                      f"task {100 * sum(row['scores'].values()) / len(row['scores']):5.1f}%  {100 * row['usd']:.3f}¢  {row['seconds']}s", flush=True)
                if spent > args.max_usd:
                    raise SystemExit(f"stopped: spent ${spent:.2f}")
    print(f"spent ${spent:.3f} this run")
    await jev.aclose(); await llm.aclose()


def cmd_report(_args) -> None:
    rows = [json.loads(l) for l in OUT.read_text().splitlines()]
    _, questions = load()
    kinds = {str(q["id"]): q["answer_type"].split(".")[-1] for q in questions}
    summary: dict = {"arms": {}, "tuning_lengths": list(TUNE_LENGTHS)}
    for arm in dict.fromkeys(r["arm"] for r in rows):
        by_len = {}
        for length in sorted({r["context_len"] for r in rows}):
            group = [r for r in rows if r["arm"] == arm and r["context_len"] == length]
            if not group:
                continue
            scores = {k: v for r in group for k, v in r["scores"].items()}
            by_kind = {kind: [v for k, v in scores.items() if kinds[k] == kind] for kind in sorted(set(kinds.values()))}
            live = [r["seconds"] for r in group if r["live"]]
            by_len[length] = {
                "n_questions": len(scores), "score": sum(scores.values()) / len(scores),
                "score_by_answer_type": {k: sum(v) / len(v) for k, v in by_kind.items() if v},
                "record_accuracy": sum(r["record_accuracy"] for r in group) / len(group),
                "usd_per_window": sum(r["usd"] for r in group) / len(group),
                "usd_per_question": sum(r["usd"] for r in group) / len(scores),
                "seconds_per_window": sum(live) / len(live) if live else None,
                "requests_per_window": sum(r["requests"] for r in group) / len(group),
                "tokens_per_window": sum(r["tokens"] for r in group) / len(group),
                "batch": group[0]["batch"],
            }
        summary["arms"][arm] = by_len
    SUMMARY.write_text(json.dumps(summary, indent=1))
    lengths = sorted({r["context_len"] for r in rows})
    print(f"{'task score by context length':<32}" + "".join(f"{l // 1024:>7}K" for l in lengths))
    for arm, by_len in summary["arms"].items():
        print(f"{arm:<32}" + "".join(f"{100 * by_len[l]['score']:>7.1f}%" if l in by_len else f"{'':>8}" for l in lengths))
    print(f"\n{'at 131K':<32}{'task':>8}{'records':>9}{'compare':>9}{'label':>8}{'number':>8}{'user':>7}{'¢/window':>10}{'¢/question':>12}{'sec':>7}{'requests':>10}")
    for arm, by_len in summary["arms"].items():
        s = by_len.get(131072)
        if s:
            t = s["score_by_answer_type"]
            print(f"{arm:<32}{100 * s['score']:>7.1f}%{100 * s['record_accuracy']:>8.1f}%{100 * t.get('COMPARISON', 0):>8.1f}%{100 * t.get('LABEL', 0):>7.1f}%"
                  f"{100 * t.get('NUMERIC', 0):>7.1f}%{100 * t.get('USER', 0):>6.0f}%{100 * s['usd_per_window']:>9.2f}¢{100 * s['usd_per_question']:>11.3f}¢"
                  f"{(format(s['seconds_per_window'], '.1f') if s['seconds_per_window'] else '-'):>7}{s['requests_per_window']:>10.0f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    tune = sub.add_parser("tune")
    tune.add_argument("--batches", type=int, nargs="+", default=[1, 10, 25, 50, 100])
    tune.add_argument("--llm-batches", type=int, nargs="*", default=[50, 200])
    tune.add_argument("--concurrency", type=int, default=64)
    run = sub.add_parser("run")
    run.add_argument("--batch", type=int, required=True)
    run.add_argument("--llm-batch", type=int, required=True)
    run.add_argument("--concurrency", type=int, default=64)
    run.add_argument("--max-usd", type=float, default=1.5)
    run.add_argument("--costly-max-len", type=int, default=131072)
    sub.add_parser("report")
    sub.add_parser("fetch")
    args = parser.parse_args()
    if args.cmd in ("report", "fetch"):
        {"report": cmd_report, "fetch": cmd_fetch}[args.cmd](args)
    else:
        asyncio.run({"tune": cmd_tune, "run": cmd_run}[args.cmd](args))


if __name__ == "__main__":
    main()
