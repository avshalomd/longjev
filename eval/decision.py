"""Kill-test: a long-input task that can be decided chunk by chunk.

Task: does this contract contain clause type X? (CUAD, contracts too long for Jev's window.)

Arms, all on the same contracts and clause types:
  jev_max        Jev answers every clause question on every chunk; the contract's score is the highest chunk score. No final call.
  embed_4k/16k   embeddings pick chunks per clause type, then one Jev call per clause type
  truncate_16k   keep the start and end, one Jev call
  jev_select_8k  longjev v1: Jev's chunk scores pick the chunks, then one Jev call per clause type
  llm:<model>    an LLM reads the whole contract once and answers every clause type as JSON

  uv run python -m eval.decision run --contracts 40 --max-usd 2
  uv run python -m eval.decision report
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from eval import cuad
from eval.report import mcnemar, wilson
from longjev import Cache, LongJev, TokenEstimator
from longjev.assembler import select_round_robin, select_truncate
from longjev.selectors import score_embed
from longjev.transport import OpenRouterTransport, TransportError

MODEL = "typesafe/jev-1.13"
JEV_USD, EMBED_USD = 0.042e-6, 0.01e-6
LLMS = {  # OpenRouter list prices per token, read 2026-09-18
    "qwen/qwen3-235b-a22b-2507": (0.087e-6, 0.35e-6),
    "google/gemini-2.5-flash": (0.30e-6, 2.50e-6),
}
OUT = Path("results/raw/decision.jsonl")
SUMMARY = Path("results/decision_summary.json")
WHOLE = Path("results/raw/decision_whole.jsonl")
WINDOW = 32_000  # Jev's limit for the state plus one question, in Jev's own tokens


async def jev_chunks(lj: LongJev, chunks, questions: dict, concurrency: int):
    """Every clause question on every chunk. Returns (probabilities per key, measured tokens per chunk)."""
    raw = {k: q.to_json() for k, q in questions.items()}
    gate = asyncio.Semaphore(concurrency)
    probs = {k: np.zeros(len(chunks), dtype=np.float32) for k in questions}
    used = np.zeros(len(chunks), dtype=np.int64)

    async def one(chunk):
        async with gate:
            data = await lj.transport.decide(chunk.text, raw, MODEL)
        for k in questions:
            probs[k][chunk.id] = float((data["answers"].get(k) or {}).get("noul", 0.0))
        used[chunk.id] = int((data.get("usage") or {}).get("input_tokens", 0) or 0)

    async def probe():
        async with gate:
            data = await lj.transport.decide("-", raw, MODEL)
        return int((data.get("usage") or {}).get("input_tokens", 0) or 0)

    question_tokens, *_ = await asyncio.gather(probe(), *(one(c) for c in chunks))
    return probs, np.maximum(1, used - question_tokens + 1), int(used.sum())


async def final_calls(lj: LongJev, chunks, questions: dict, rel: dict, budget: int) -> dict[str, float]:
    """One Jev call per clause type over the chunks picked for that clause type."""
    async def one(k):
        q = {k: questions[k]}
        kept = select_round_robin(chunks, {k: rel[k]}, lj.effective_budget(q, budget))
        answers, _, _ = await lj.judge(chunks, q, kept)
        return k, float(answers[k]["noul"])
    return dict(await asyncio.gather(*(one(k) for k in questions)))


async def ask_llm(llm: OpenRouterTransport, model: str, context: str, definitions: dict) -> dict:
    listing = "\n".join(f'- "{k}": {name}. {details}' for k, (name, details) in definitions.items())
    body = {
        "model": model, "temperature": 0, "max_tokens": 400, "usage": {"include": True},
        "messages": [
            {"role": "system", "content": "You review contracts. Answer only with a JSON object."},
            {"role": "user", "content": (
                f"<contract>\n{context}\n</contract>\n\nFor each clause type below, does the contract contain such a clause?\n"
                f"{listing}\n\nReply with one JSON object mapping each key to true or false. No other text."
            )},
        ],
    }
    if "gemini" in model:
        body["reasoning"] = {"enabled": False}
    started = time.perf_counter()
    data = await llm._post("/v1/chat/completions", body)
    seconds = time.perf_counter() - started
    text = (data["choices"][0]["message"].get("content") or "")
    try:
        parsed = json.loads(text[text.index("{"): text.rindex("}") + 1])
    except ValueError:
        parsed = {}
    usage = data.get("usage") or {}
    price_in, price_out = LLMS[model]
    cost = usage.get("cost")
    if cost is None:
        cost = usage.get("prompt_tokens", 0) * price_in + usage.get("completion_tokens", 0) * price_out
    return {
        "answers": {k: (bool(parsed[k]) if k in parsed else None) for k in definitions},
        "seconds": round(seconds, 2), "usd": float(cost),
        "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens"),
    }


async def run_contract(item: dict, definitions: dict, lj: LongJev, llm: OpenRouterTransport, concurrency: int) -> dict:
    chunk_qs = {k: cuad.chunk_question(*d) for k, d in definitions.items()}
    doc_qs = {k: cuad.contract_question(*d) for k, d in definitions.items()}
    usage = lj.transport.usage
    record = {"id": item["id"], "labels": item["labels"], "chars": len(item["context"]), "arms": {}}

    def snapshot():
        return usage.jev_input_tokens, usage.embed_tokens, usage.jev_requests + usage.embed_requests, usage.cache_hits

    def arm(name, scores, before, seconds, extra_usd=0.0, extra_seconds=0.0, allow_hits=0):
        after = snapshot()
        record["arms"][name] = {
            "scores": {k: round(float(v), 4) for k, v in scores.items()},
            "usd": (after[0] - before[0]) * JEV_USD + (after[1] - before[1]) * EMBED_USD + extra_usd,
            "seconds": round(seconds + extra_seconds, 2), "requests": after[2] - before[2],
            "live": after[3] - before[3] <= allow_hits,
        }

    chunks = lj.chunk(item["context"])
    before, started = snapshot(), time.perf_counter()
    probs, measured, _ = await jev_chunks(lj, chunks, chunk_qs, concurrency)
    jev_seconds = time.perf_counter() - started
    # the probe request is the same for every contract, so one cache hit is expected
    arm("jev_max", {k: v.max() for k, v in probs.items()}, before, jev_seconds, allow_hits=1)
    jev_usd, jev_live = record["arms"]["jev_max"]["usd"], record["arms"]["jev_max"]["live"]
    chunks = [replace(c, est_tokens=int(measured[c.id])) for c in chunks]
    record["n_chunks"], record["tokens"] = len(chunks), int(measured.sum())
    record["top_chunk"] = {k: int(v.argmax()) for k, v in probs.items()}

    before, started = snapshot(), time.perf_counter()
    embedded = await score_embed(chunks, doc_qs, lj.transport, lj.embed_model)
    embed_seconds, embed_before = time.perf_counter() - started, before
    embed_usd = (snapshot()[1] - before[1]) * EMBED_USD
    embed_live = snapshot()[3] == before[3]
    for budget in (4000, 16000):
        before, started = snapshot(), time.perf_counter()
        scores = await final_calls(lj, chunks, doc_qs, embedded.rel, budget)
        arm(f"embed_{budget // 1000}k", scores, before, time.perf_counter() - started, embed_usd, embed_seconds)
        record["arms"][f"embed_{budget // 1000}k"]["live"] &= embed_live

    before, started = snapshot(), time.perf_counter()
    kept = select_truncate(chunks, lj.effective_budget(doc_qs, 16000))
    answers, _, _ = await lj.judge(chunks, doc_qs, kept)
    arm("truncate_16k", {k: answers[k]["noul"] for k in doc_qs}, before, time.perf_counter() - started)

    before, started = snapshot(), time.perf_counter()
    scores = await final_calls(lj, chunks, doc_qs, probs, 8000)
    arm("jev_select_8k", scores, before, time.perf_counter() - started, jev_usd, jev_seconds)
    record["arms"]["jev_select_8k"]["live"] &= jev_live

    results = await asyncio.gather(*(ask_llm(llm, m, item["context"], definitions) for m in LLMS), return_exceptions=True)
    for model, out in zip(LLMS, results):
        if isinstance(out, Exception):
            print(f"  ! {model}: {str(out)[:160]}", flush=True)
            continue
        record["arms"]["llm:" + model] = {
            "scores": {k: (None if v is None else float(v)) for k, v in out["answers"].items()},
            "usd": out["usd"], "seconds": out["seconds"], "requests": 1, "live": True,
            "prompt_tokens": out["prompt_tokens"], "completion_tokens": out["completion_tokens"],
        }
    return record


async def cmd_run(args) -> None:
    contracts, definitions = cuad.load(args.contracts)
    done = {}
    if OUT.exists():
        done = {r["id"]: r for r in map(json.loads, filter(str.strip, OUT.read_text().splitlines()))}
    lj = LongJev(model=MODEL, concurrency=args.concurrency, cache=Cache(".longjev_cache/bench.sqlite"), estimator=TokenEstimator(3.0))
    llm = OpenRouterTransport(timeout=600.0)
    spent = llm_spent = 0.0
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("a") as handle:
        for n, item in enumerate(contracts, 1):
            if item["id"] in done:
                continue
            if spent >= args.max_usd:
                print(f"stopping at the ${args.max_usd} guard", flush=True)
                break
            record = await run_contract(item, definitions, lj, llm, args.concurrency)
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            llm_spent += sum(a["usd"] for name, a in record["arms"].items() if name.startswith("llm:"))
            spent = lj.transport.usage.usd + llm_spent
            print(f"  {n}/{len(contracts)}  {record['tokens']:>7} tok  {record['n_chunks']:>3} chunks  "
                  f"jev_max {record['arms']['jev_max']['seconds']}s  spent ${spent:.3f}", flush=True)
    await lj.aclose()
    await llm.aclose()
    print(f"spent ${spent:.3f} this run (Jev + embeddings ${lj.transport.usage.usd:.3f})")


async def cmd_whole(args) -> None:
    """Control: contracts that fit Jev's window, read whole in one call with all ten questions.
    Needs the contract sizes that `run` measured."""
    contracts, definitions = cuad.load(args.contracts)
    doc_qs = {k: cuad.contract_question(*d).to_json() for k, d in definitions.items()}
    sizes = {r["id"]: r["tokens"] for r in map(json.loads, filter(str.strip, OUT.read_text().splitlines()))}
    lj = LongJev(model=MODEL, cache=Cache(".longjev_cache/bench.sqlite"), estimator=TokenEstimator(3.0))
    usage = lj.transport.usage
    with WHOLE.open("w") as handle:
        for item in contracts:
            if sizes.get(item["id"], WINDOW + 1) > WINDOW:
                continue
            tokens, hits, started = usage.jev_input_tokens, usage.cache_hits, time.perf_counter()
            try:
                data = await lj.transport.decide(item["context"], doc_qs, MODEL)
            except TransportError as exc:
                if not exc.too_long:
                    raise
                print(f"  {item['id'][:40]}: too long with its questions, skipped", flush=True)
                continue
            arm = {"scores": {k: round(float((data["answers"].get(k) or {}).get("noul", 0.0)), 4) for k in doc_qs},
                   "usd": (usage.jev_input_tokens - tokens) * JEV_USD, "seconds": round(time.perf_counter() - started, 2),
                   "requests": 1, "live": usage.cache_hits == hits}
            handle.write(json.dumps({"id": item["id"], "arm": arm}) + "\n")
            handle.flush()
            print(f"  {sizes[item['id']]:>7} tok  {arm['seconds']}s  spent ${usage.usd:.4f}", flush=True)
    await lj.aclose()


async def cmd_chunks(args) -> None:
    """Same contracts, Jev on every chunk, at larger chunk sizes: does accuracy hold when the
    questions are repeated less often?"""
    contracts, definitions = cuad.load(args.contracts)
    chunk_qs = {k: cuad.chunk_question(*d) for k, d in definitions.items()}
    out = Path("results/raw/decision_chunks.jsonl")
    with out.open("w") as handle:
        for size in args.sizes:
            lj = LongJev(model=MODEL, chunk_tokens=size, concurrency=args.concurrency,
                         cache=Cache(".longjev_cache/bench.sqlite"), estimator=TokenEstimator(3.0))
            usage = lj.transport.usage
            for item in contracts:
                chunks = lj.chunk(item["context"])
                tokens_before, hits_before, started = usage.jev_input_tokens, usage.cache_hits, time.perf_counter()
                probs, measured, _ = await jev_chunks(lj, chunks, chunk_qs, args.concurrency)
                handle.write(json.dumps({
                    "id": item["id"], "size": size, "n_chunks": len(chunks), "labels": item["labels"],
                    "scores": {k: round(float(v.max()), 4) for k, v in probs.items()},
                    "usd": (usage.jev_input_tokens - tokens_before) * JEV_USD, "tokens": int(measured.sum()),
                    "seconds": round(time.perf_counter() - started, 2), "live": usage.cache_hits - hits_before <= 1,
                }) + "\n")
                handle.flush()
            print(f"  chunk size {size}: done, spent ${usage.usd:.3f}", flush=True)
            await lj.aclose()
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    print(f"{'chunk tokens':>12}{'chunks':>8}{'acc':>8}{'F1':>7}{'prec':>7}{'rec':>7}{'AUROC':>7}{'¢/contract':>12}{'x tokens':>10}{'sec':>7}")
    for size in args.sizes:
        group = [r for r in rows if r["size"] == size]
        y = np.array([t for r in group for t in r["labels"].values()])
        s = np.array([r["scores"][k] for r in group for k in r["labels"]])
        pred = s >= 0.5
        tp, fp, fn = int((pred & y).sum()), int((pred & ~y).sum()), int((~pred & y).sum())
        fresh = [r for r in group if r["live"]] or group  # a cache hit bills nothing, so it is not a price
        sent = sum(r["usd"] for r in fresh) / JEV_USD / sum(r["tokens"] for r in fresh)
        live = [r["seconds"] for r in group if r["live"]]
        print(f"{size:>12}{np.median([r['n_chunks'] for r in group]):>8.0f}{100 * (pred == y).mean():>7.1f}%{2 * tp / max(1, 2 * tp + fp + fn):>7.2f}"
              f"{tp / max(1, tp + fp):>7.2f}{tp / max(1, tp + fn):>7.2f}{auroc(list(y), list(s)):>7.2f}"
              f"{100 * np.mean([r['usd'] for r in fresh]):>11.2f}¢{sent:>10.2f}{(np.median(live) if live else float('nan')):>7.1f}")


def auroc(labels: list[bool], scores: list[float]) -> float | None:
    y, s = np.asarray(labels, dtype=bool), np.asarray(scores, dtype=float)
    if y.all() or not y.any():
        return None
    order = s.argsort()
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    for v in np.unique(s):  # average ranks over ties
        tie = s == v
        ranks[tie] = ranks[tie].mean()
    pos = int(y.sum())
    return float((ranks[y].sum() - pos * (pos + 1) / 2) / (pos * (len(y) - pos)))


def decisions(rows: list[dict], name: str) -> tuple[list[bool], list[float], int]:
    """Truth and score for every (contract, clause type) of one arm. An unparsed answer counts as wrong."""
    labels, scores, missing = [], [], 0
    for r in rows:
        a = r["arms"].get(name)
        if not a:
            continue
        for k, truth in r["labels"].items():
            v = a["scores"].get(k)
            if v is None:
                missing += 1
                v = 0.0 if truth else 1.0
            labels.append(truth)
            scores.append(v)
    return labels, scores, missing


def paired(rows: list[dict], name: str, other: str) -> dict:
    """Exact McNemar on the decisions both arms made. Decisions from one contract are not
    independent, so read the p-value as optimistic."""
    both = [r for r in rows if name in r["arms"] and other in r["arms"]]
    (y, a, _), (_, b, _) = decisions(both, name), decisions(both, other)
    right_a, right_b = (np.asarray(a) >= 0.5) == np.asarray(y), (np.asarray(b) >= 0.5) == np.asarray(y)
    only_a, only_b = int((right_a & ~right_b).sum()), int((~right_a & right_b).sum())
    return {"only_this": only_a, "only_other": only_b, "p": mcnemar(only_a, only_b)}


def cmd_report(_args) -> None:
    rows = [json.loads(l) for l in OUT.read_text().splitlines() if l.strip()]
    if WHOLE.exists():  # the whole-contract control, for the contracts that fit
        for w in map(json.loads, filter(str.strip, WHOLE.read_text().splitlines())):
            next(r for r in rows if r["id"] == w["id"])["arms"]["jev_whole"] = w["arm"]
    arms = list(dict.fromkeys(name for r in rows for name in r["arms"]))
    summary = {"n_contracts": len(rows), "n_decisions": 0, "tokens_mean": float(np.mean([r["tokens"] for r in rows])),
               "tokens_max": max(r["tokens"] for r in rows), "chunks_median": float(np.median([r["n_chunks"] for r in rows])), "arms": {}}
    over = [r for r in rows if r["tokens"] > WINDOW]
    for name in arms:
        labels, scores, missing = decisions(rows, name)
        y, pred = np.asarray(labels), np.asarray(scores) >= 0.5
        tp, fp, fn = int((pred & y).sum()), int((pred & ~y).sum()), int((~pred & y).sum())
        correct = int((pred == y).sum())
        low, high = wilson(correct, len(y))
        with_arm = [r for r in rows if name in r["arms"]]
        live = [r["arms"][name]["seconds"] for r in with_arm if r["arms"][name]["live"]]
        fresh = [r for r in with_arm if r["arms"][name]["live"]] or with_arm  # cache hits bill nothing
        y_over, s_over, _ = decisions(over, name)
        summary["arms"][name] = {
            "n": len(y), "acc": correct / len(y), "low": low, "high": high,
            "precision": tp / max(1, tp + fp), "recall": tp / max(1, tp + fn),
            "f1": 2 * tp / max(1, 2 * tp + fp + fn), "auroc": None if name.startswith("llm:") else auroc(labels, scores),
            "said_yes": float(pred.mean()), "missing": missing,
            "usd_per_contract": float(np.mean([r["arms"][name]["usd"] for r in fresh])),
            "vs_jev_max": paired(rows, "jev_max", name) if name != "jev_max" else None,
            "over_window": {"n": len(y_over), "acc": float(((np.asarray(s_over) >= 0.5) == np.asarray(y_over)).mean()),
                            "vs_jev_max": paired(over, "jev_max", name) if name != "jev_max" else None} if y_over else None,
            "seconds_median": float(np.median(live)) if live else None, "seconds_max": max(live) if live else None,
            "n_live": len(live), "requests_median": float(np.median([r["arms"][name]["requests"] for r in with_arm])),
        }
        summary["n_decisions"] = max(summary["n_decisions"], len(y))
    summary["base_rate"] = float(np.mean([v for r in rows for v in r["labels"].values()]))
    summary["always_no_acc"] = 1 - summary["base_rate"]
    summary["tokens_min"] = min(r["tokens"] for r in rows)
    summary["n_over_window"] = len(over)
    SUMMARY.write_text(json.dumps(summary, indent=1))
    print(f"{len(rows)} contracts ({len(over)} over Jev's {WINDOW:,}-token window), {summary['n_decisions']} decisions, "
          f"{100 * summary['base_rate']:.0f}% present (always-no scores {100 * summary['always_no_acc']:.0f}%), "
          f"{summary['tokens_min'] / 1000:.0f}K to {summary['tokens_max'] / 1000:.0f}K tokens, mean {summary['tokens_mean'] / 1000:.0f}K")
    print(f"{'arm':<34}{'n':>5}{'acc':>7}{'F1':>7}{'prec':>7}{'rec':>7}{'AUROC':>7}{'¢/contract':>12}{'sec':>7}{'req':>6}"
          f"{'p vs jev_max':>14}{'acc >window':>13}{'p':>7}")
    for name, a in summary["arms"].items():
        test, big = a["vs_jev_max"], a["over_window"]
        print(f"{name:<34}{a['n']:>5}{100 * a['acc']:>6.1f}%{a['f1']:>7.2f}{a['precision']:>7.2f}{a['recall']:>7.2f}"
              f"{(a['auroc'] if a['auroc'] is not None else float('nan')):>7.2f}{100 * a['usd_per_contract']:>11.2f}¢"
              f"{(a['seconds_median'] or float('nan')):>7.1f}{a['requests_median']:>6.0f}"
              f"{(test['p'] if test else float('nan')):>14.3f}{(100 * big['acc'] if big else float('nan')):>12.1f}%"
              f"{(big['vs_jev_max']['p'] if big and big['vs_jev_max'] else float('nan')):>7.3f}")


if __name__ == "__main__":
    load_dotenv(".env")
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run")
    run.add_argument("--contracts", type=int, default=40)
    run.add_argument("--max-usd", type=float, default=2.0)
    run.add_argument("--concurrency", type=int, default=64)
    sub.add_parser("report")
    sizes = sub.add_parser("chunks")
    sizes.add_argument("--contracts", type=int, default=40)
    sizes.add_argument("--concurrency", type=int, default=64)
    sizes.add_argument("--sizes", type=int, nargs="+", default=[2000, 4000, 8000])
    whole = sub.add_parser("whole")
    whole.add_argument("--contracts", type=int, default=40)
    args = parser.parse_args()
    if args.cmd == "run":
        asyncio.run(cmd_run(args))
    elif args.cmd == "whole":
        asyncio.run(cmd_whole(args))
    elif args.cmd == "chunks":
        asyncio.run(cmd_chunks(args))
    else:
        cmd_report(args)
