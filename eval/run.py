"""LongBench v2 benchmark for longjev.

  uv run python -m eval.run calibrate
  uv run python -m eval.run run --max-context-tokens 120000 --max-usd 3
  uv run python -m eval.run report
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

from eval.longbench import load, question
from longjev import Cache, LongJev, TokenEstimator
from longjev.reducer import vote
from longjev.selectors import combine_shortlist, score_embed, score_jev, shortlist

MODEL = "typesafe/jev-1.13"
BUDGETS = (8_000, 16_000, 28_000)
CHARS_PER_TOKEN = 3.0  # `calibrate` measured 3.08 on average, 2.4 to 3.5 by item
RAW = Path("results/raw")
ITEMS_OUT = RAW / "items.jsonl"


class Spend:
    def __init__(self, lj: LongJev, cap: float):
        self.lj, self.cap = lj, cap

    def check(self) -> None:
        usd = self.lj.transport.usage.usd
        if usd > self.cap:
            raise SystemExit(f"spend cap reached: ${usd:.2f} > ${self.cap:.2f}")


def make(concurrency: int) -> LongJev:
    return LongJev(
        model=MODEL, concurrency=concurrency, cache=Cache(".longjev_cache/bench.sqlite"),
        estimator=TokenEstimator(CHARS_PER_TOKEN), score_log=RAW / "scores.jsonl",
    )


def compact(answer: dict | None) -> dict | None:
    if not answer:
        return None
    return {"choice": answer.get("choice"), "probs": answer.get("probabilities"), "conf": answer.get("confidence")}


async def judge_at(lj: LongJev, chunks, qs, scores, budget: int) -> dict:
    budget = lj.effective_budget(qs, budget)
    kept = lj.select(chunks, scores, budget)
    started = time.perf_counter()
    answers, suff, kept = await lj.judge(chunks, qs, kept)
    return {
        **compact(answers["q"]), "suff": suff["q"], "n_kept": len(kept), "kept": sorted(kept),
        "state_tokens": sum(chunks[i].est_tokens for i in kept),
        "seconds": round(time.perf_counter() - started, 3),
    }


def budgets_for(item: dict, fits: bool, frozen: dict | None) -> dict[str, tuple[int, ...]]:
    """Dev items sweep every budget. Test items run only at the budgets frozen on dev.
    Items that already fit are the control group: compressed to the smallest budget."""
    names = ("truncate", "embed", "jev", "embed_then_jev")
    if fits:
        return {n: BUDGETS[:1] for n in names}
    if item["split"] == "dev" or not frozen:
        return {n: BUDGETS for n in names}
    return {n: (int(frozen[n]),) for n in names}


async def run_item(lj: LongJev, item: dict, gate: asyncio.Semaphore, previous: dict | None, frozen: dict | None = None) -> dict:
    qs = question(item)
    chunks = lj.chunk(item["context"])
    usage = lj.transport.usage
    record = {
        "id": item["_id"], "split": item["split"], "domain": item["domain"],
        "sub_domain": item["sub_domain"], "difficulty": item["difficulty"],
        "length": item["length"], "answer": item["answer"], "n_chunks": len(chunks),
        "context_tokens": sum(c.est_tokens for c in chunks), "results": {},
    }

    t0, r0 = time.perf_counter(), usage.embed_requests
    embedded = await score_embed(chunks, qs, lj.transport, lj.embed_model)
    embed_time = {"seconds": round(time.perf_counter() - t0, 2), "requests": usage.embed_requests - r0}

    t0, r0, k0 = time.perf_counter(), usage.jev_requests, usage.jev_input_tokens
    jev = await score_jev(chunks, qs, lj.transport, lj.model, lj.estimator, gate=gate)
    jev_time = {
        "seconds": round(time.perf_counter() - t0, 2), "requests": usage.jev_requests - r0,
        "tokens": usage.jev_input_tokens - k0,
    }
    lj._log_scores(item["context"], chunks, jev, doc_id=item["_id"])
    chunks = lj.recalibrate(chunks, jev)  # budgets below are in tokens as Jev counted them
    record["chars_per_token"] = jev.chars_per_token
    record["context_tokens"] = sum(c.est_tokens for c in chunks)

    # timings only mean something when the requests were real, not cache hits
    old = (previous or {}).get("timing", {})
    live = jev_time["requests"] > 0.5 * len(chunks)
    record["timing"] = {
        "embed": embed_time if embed_time["requests"] else old.get("embed", embed_time),
        "jev": jev_time if live else old.get("jev", {"seconds": None, "requests": 0, "tokens": 0}),
    }
    record["jev_scoring_tokens"] = jev.input_tokens
    record["failed_chunks"] = len(jev.failed)
    if len(chunks) <= 800:
        record["strip"] = {
            "jev": [round(float(x), 2) for x in jev.rel["q"]],
            "embed": [round(float(x), 3) for x in embedded.rel["q"]],
        }
    ids = shortlist(embedded.rel)
    record["shortlist"] = len(ids)
    scores = {
        "truncate": None, "embed": embedded, "jev": jev,
        "embed_then_jev": combine_shortlist(embedded, jev, ids),
    }
    record["vote"] = compact(vote(jev, qs)["q"])
    record["vote_shortlist"] = compact(vote(scores["embed_then_jev"], qs)["q"])

    fits = record["context_tokens"] + 600 <= lj.max_budget_tokens
    record["fits"] = fits
    if fits:
        answers, suff, _ = await lj.judge(chunks, qs, list(range(len(chunks))))
        record["direct"] = {**compact(answers["q"]), "suff": suff["q"]}
    plan = budgets_for(item, fits, frozen)
    for name, sc in scores.items():
        for budget in plan[name]:
            if fits and budget >= record["context_tokens"]:
                continue  # nothing to compress at this budget
            record["results"][f"{name}@{budget}"] = await judge_at(lj, chunks, qs, sc, budget)
    return record


async def cmd_run(args) -> None:
    items = load()
    done = {}
    if ITEMS_OUT.exists():
        for line in ITEMS_OUT.read_text().splitlines():
            row = json.loads(line)
            done[row["id"]] = row
    est = TokenEstimator(CHARS_PER_TOKEN)
    chosen = [i for i in items if est.estimate(i["context"]) <= args.max_context_tokens]
    if args.split != "all":
        chosen = [i for i in chosen if i["split"] == args.split]
    if args.length:
        chosen = [i for i in chosen if i["length"] == args.length]
    chosen.sort(key=lambda i: len(i["context"]))
    if args.limit:
        chosen = chosen[: args.limit]
    print(f"{len(chosen)} items, {sum(est.estimate(i['context']) for i in chosen)/1e6:.1f}M context tokens")

    frozen = None
    if args.split == "test":
        frozen = json.loads(Path("results/summary.json").read_text())["frozen_budgets"]
        print("frozen budgets from dev:", frozen)
    lj = make(args.concurrency)
    spend = Spend(lj, args.max_usd)
    gate = asyncio.Semaphore(args.concurrency)
    item_gate = asyncio.Semaphore(args.items_at_once)
    RAW.mkdir(parents=True, exist_ok=True)
    finished = 0
    started = time.perf_counter()

    async def one(item):
        nonlocal finished
        async with item_gate:
            spend.check()
            try:
                record = await run_item(lj, item, gate, done.get(item["_id"]), frozen)
            except Exception as exc:  # keep going; report at the end
                print(f"  ! {item['_id']}: {type(exc).__name__}: {str(exc)[:200]}", flush=True)
                return
            done[item["_id"]] = record
            finished += 1
            u = lj.transport.usage
            if finished % 5 == 0 or finished == len(chosen):
                print(
                    f"  {finished}/{len(chosen)}  ${u.usd:.3f}  jev_req={u.jev_requests} "
                    f"cache_hits={u.cache_hits}  {time.perf_counter()-started:.0f}s", flush=True,
                )
                save(done)

    try:
        await asyncio.gather(*(one(i) for i in chosen))
    finally:
        save(done)
        u = lj.transport.usage
        print(f"spent ${u.usd:.3f}: jev {u.jev_input_tokens/1e6:.2f}M tok in {u.jev_requests} req, "
              f"embed {u.embed_tokens/1e6:.2f}M tok; observed chars/token {lj.estimator.observed_chars_per_token}")
        await lj.aclose()


def save(done: dict) -> None:
    tmp = ITEMS_OUT.with_suffix(".tmp")
    tmp.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in done.values()) + "\n")
    tmp.replace(ITEMS_OUT)


async def cmd_calibrate(args) -> None:
    """Measures characters per Jev token on real benchmark text."""
    items = load()[:: max(1, 503 // 8)][:8]
    lj = LongJev(model=MODEL)
    total_chars = total_tokens = 0
    for item in items:
        text = item["context"][:6000]
        data = await lj.transport.decide(text, {"q": {"type": "noul", "instructions": "Is this text in English?"}}, MODEL)
        used = data["usage"]["input_tokens"]
        print(item["domain"][:28].ljust(28), len(text), used, round(len(text) / used, 2), data["answers"]["q"])
        total_chars += len(text)
        total_tokens += used
    print("chars per token:", round(total_chars / total_tokens, 3), "| raw response keys:", list(data))
    await lj.aclose()


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("calibrate")
    run = sub.add_parser("run")
    run.add_argument("--max-context-tokens", type=int, default=120_000)
    run.add_argument("--split", default="all", choices=["all", "dev", "test"])
    run.add_argument("--limit", type=int, default=0)
    run.add_argument("--length", default="", choices=["", "short", "medium", "long"])
    run.add_argument("--max-usd", type=float, default=1.0)
    run.add_argument("--concurrency", type=int, default=24)
    run.add_argument("--items-at-once", type=int, default=4)
    sub.add_parser("report")
    args = parser.parse_args()
    if args.cmd == "report":
        from eval.report import build
        build()
    else:
        asyncio.run({"run": cmd_run, "calibrate": cmd_calibrate}[args.cmd](args))


if __name__ == "__main__":
    sys.exit(main())
