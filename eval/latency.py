"""End-to-end latency as a user sees it: one question at a time, no cache, the real
`system_one` call for each selector.

  uv run python -m eval.latency --items 16
  uv run python -m eval.latency --indexed     # same questions, chunk vectors served from a prebuilt index
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

from dotenv import load_dotenv

from eval.longbench import load, question
from longjev import LongJev, TokenEstimator
from longjev.cache import Cache, request_key
from longjev.transport import Transport, make_transport

MODEL = "typesafe/jev-1.13"
SELECTORS = ("truncate", "embed", "embed_then_jev", "jev")
OUT = Path("results/raw/latency.jsonl")
OUT_INDEXED = Path("results/raw/latency_indexed.jsonl")
INDEXED_SELECTORS = ("embed", "embed_then_jev")


class IndexedTransport:
    """Chunk vectors come from the benchmark cache, standing in for a prebuilt index.
    The query embedding and every Jev call are live."""

    def __init__(self, inner: Transport, cache: Cache):
        self.inner, self.cache, self.index_misses = inner, cache, 0

    @property
    def usage(self):
        return self.inner.usage

    async def decide(self, state, questions, model):
        return await self.inner.decide(state, questions, model)

    async def embed(self, texts, model):
        if texts and texts[0].startswith("Instruct: "):
            return await self.inner.embed(texts, model)
        found = [self.cache.get_vector(request_key("embed", model, t)) for t in texts]
        missing = [i for i, v in enumerate(found) if v is None]
        if missing:
            self.index_misses += len(missing)
            for i, vec in zip(missing, await self.inner.embed([texts[i] for i in missing], model)):
                found[i] = vec
        return found

    async def aclose(self):
        await self.inner.aclose()


def pick_items(n_long: int, n_fit: int) -> list[dict]:
    rows = [json.loads(l) for l in Path("results/raw/items.jsonl").read_text().splitlines() if l.strip()]
    long_rows = sorted((r for r in rows if not r["fits"] and r["split"] == "test"), key=lambda r: r["context_tokens"])
    fit_rows = sorted((r for r in rows if r["fits"] and r["split"] == "test"), key=lambda r: r["context_tokens"])
    def spread(group, n):
        if n >= len(group):
            return group
        return [group[round(i * (len(group) - 1) / (n - 1))] for i in range(n)]
    return spread(long_rows, n_long) + spread(fit_rows, n_fit)


async def main(args) -> None:
    frozen = json.loads(Path("results/summary.json").read_text())["frozen_budgets"]
    data = {i["_id"]: i for i in load()}
    if args.indexed:
        # the same questions the cold run measured, so the two are comparable
        seen = [json.loads(l) for l in OUT.read_text().splitlines() if l.strip()]
        ids = list(dict.fromkeys(x["id"] for x in seen if not x["passthrough"]))
        by_id = {json.loads(l)["id"]: json.loads(l) for l in Path("results/raw/items.jsonl").read_text().splitlines() if l.strip()}
        chosen, selectors, out = [by_id[i] for i in ids], INDEXED_SELECTORS, OUT_INDEXED
        cache = Cache(".longjev_cache/bench.sqlite")
    else:
        chosen, selectors, out = pick_items(args.items, args.fit_items), SELECTORS, OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as handle:
        for n, row in enumerate(chosen, 1):
            item = data[row["id"]]
            for selector in selectors:
                transport = IndexedTransport(make_transport(), cache) if args.indexed else None
                lj = LongJev(
                    model=MODEL, selector=selector, budget_tokens=int(frozen[selector]),
                    concurrency=args.concurrency, estimator=TokenEstimator(3.0), transport=transport,
                )
                started = time.perf_counter()
                try:
                    result = await lj.asystem_one(item["context"], question(item))
                except Exception as exc:
                    print(f"  ! {row['id']} {selector}: {str(exc)[:160]}", flush=True)
                    await lj.aclose()
                    continue
                seconds = time.perf_counter() - started
                info = result.long_input
                record = {
                    "id": row["id"], "selector": selector, "context_tokens": row["context_tokens"],
                    "n_chunks": row["n_chunks"], "fits": row["fits"], "seconds": round(seconds, 2),
                    "requests": info.n_requests, "usd": info.est_cost_usd, "passthrough": info.passthrough,
                    "widened": info.widened, "correct": result.answers["q"].choice == item["answer"],
                }
                if args.indexed:
                    record["index_misses"] = transport.index_misses
                handle.write(json.dumps(record) + "\n")
                handle.flush()
                await lj.aclose()
                if info.passthrough:
                    break  # every selector passes through the same way
            print(f"  {n}/{len(chosen)} done ({row['context_tokens']} tokens)", flush=True)
    print("latency run finished")


if __name__ == "__main__":
    load_dotenv(".env")
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", type=int, default=16)
    parser.add_argument("--fit-items", type=int, default=6)
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--indexed", action="store_true")
    asyncio.run(main(parser.parse_args()))
