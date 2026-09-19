"""LongJev: the same call as Jev's system_one, for a state of any length."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping

from .assembler import render_state, select_round_robin, select_truncate
from .cache import Cache
from .chunker import Chunk, chunk_state
from .questions import SUFF, Question, final_questions, to_json
from .reducer import agrees, vote
from .selectors import (
    SELECTORS,
    ChunkScores,
    combine_shortlist,
    score_embed,
    score_jev,
    shortlist,
    split_questions,
)
from .tokens import TokenEstimator
from .transport import CachedTransport, Transport, TransportError, make_transport

SINGLE_LIMIT_TOKENS = 32_000  # state plus the longest single question
MARGIN = 0.9


class Answer(dict):
    """A Jev answer. Fields are available as attributes: .choice, .noul, .score, ..."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None


@dataclass
class LongInputInfo:
    passthrough: bool
    selector: str | None = None
    n_chunks: int = 0
    kept: list[dict] = field(default_factory=list)  # id, label, score
    state_tokens: int = 0
    widened: bool = False
    sufficiency: dict[str, float] = field(default_factory=dict)
    vote: dict[str, dict | None] = field(default_factory=dict)
    agreement: dict[str, bool | None] = field(default_factory=dict)
    failed_chunks: list[int] = field(default_factory=list)
    n_requests: int = 0
    est_cost_usd: float = 0.0


@dataclass
class LongJevResult:
    model: str
    answers: dict[str, Answer]
    usage: dict[str, int]
    long_input: LongInputInfo


class LongJev:
    def __init__(
        self,
        model: str = "typesafe/jev-1.13",  # the one name OpenRouter accepts
        selector: str = "jev",
        budget_tokens: int = 16_000,
        max_budget_tokens: int = 28_000,
        chunk_tokens: int = 800,
        concurrency: int = 16,
        cache: str | Path | Cache | None = None,
        transport: Transport | None = None,
        embed_model: str = "qwen/qwen3-embedding-8b",
        score_log: str | Path | None = None,
        estimator: TokenEstimator | None = None,
        provider: str | None = None,
    ):
        if selector not in SELECTORS:
            raise ValueError(f"selector must be one of {SELECTORS}")
        self.model = model
        self.selector = selector
        self.budget_tokens = budget_tokens
        self.max_budget_tokens = max_budget_tokens
        self.chunk_tokens = chunk_tokens
        self.concurrency = concurrency
        self.embed_model = embed_model
        self.estimator = estimator or TokenEstimator()
        self.score_log = Path(score_log) if score_log else None
        if cache is not None and not isinstance(cache, Cache):
            cache = Cache(cache)  # before the transport, so a bad path leaves no client open
        transport = transport or make_transport(provider)
        if cache is not None:
            transport = CachedTransport(transport, cache)
        self.transport = transport
        self._loop: asyncio.AbstractEventLoop | None = None  # system_one's own loop, kept between calls

    # ---- steps (the benchmark calls these directly) ----

    def chunk(self, state: Any) -> list[Chunk]:
        return chunk_state(state, self.chunk_tokens, self.estimator)

    async def score(
        self,
        chunks: list[Chunk],
        questions: Mapping[str, Question],
        selector: str | None = None,
        on_done=None,
    ) -> ChunkScores | None:
        selector = selector or self.selector
        if selector == "truncate":
            return None
        if selector == "embed":
            return await score_embed(chunks, questions, self.transport, self.embed_model)
        if selector == "jev":
            scores = await score_jev(
                chunks, questions, self.transport, self.model, self.estimator,
                self.concurrency, on_done=on_done,
            )
        else:
            embedded = await score_embed(chunks, questions, self.transport, self.embed_model)
            ids = shortlist(embedded.rel)
            jev = await score_jev(
                chunks, questions, self.transport, self.model, self.estimator,
                self.concurrency, only=ids, on_done=on_done,
            )
            scores = combine_shortlist(embedded, jev, ids)
        if len(scores.failed) > 0.1 * max(1, len(chunks)):
            raise RuntimeError(f"{len(scores.failed)} of {len(chunks)} chunk requests failed")
        return scores

    @staticmethod
    def recalibrate(chunks: list[Chunk], scores: ChunkScores | None) -> list[Chunk]:
        """Chunk sizes as Jev counted them during scoring. Chunks Jev did not read
        use the characters-per-token ratio measured on the ones it did."""
        if scores is None or not scores.chars_per_token:
            return chunks
        out = []
        for c in chunks:
            measured = int(scores.tokens[c.id]) if scores.tokens is not None else 0
            size = measured or math.ceil(1.05 * len(c.text) / scores.chars_per_token)
            out.append(replace(c, est_tokens=size))
        return out

    def select(self, chunks: list[Chunk], scores: ChunkScores | None, budget_tokens: int) -> list[int]:
        if scores is None:
            return select_truncate(chunks, budget_tokens)
        return select_round_robin(chunks, scores.rel, budget_tokens)

    def effective_budget(self, questions: Mapping[str, Question], budget_tokens: int) -> int:
        longest = max(self.estimator.estimate(q) for q in final_questions(questions).values())
        return max(1_000, min(budget_tokens, int(SINGLE_LIMIT_TOKENS * MARGIN) - longest))

    async def judge(
        self, chunks: list[Chunk], questions: Mapping[str, Question], kept: list[int]
    ) -> tuple[dict[str, dict], dict[str, float], list[int]]:
        """Final judgment over the kept chunks. `kept` is in selection order, so the
        tail can be dropped if Jev says the request is too long."""
        asked = final_questions(questions)
        kept = list(kept)
        for attempt in range(4):
            state = render_state(chunks, kept)
            try:
                answers: dict[str, dict] = {}
                size = sum(chunks[i].est_tokens for i in kept)
                for group in split_questions(asked, size, self.estimator):
                    data = await self.transport.decide(state, group, self.model)
                    answers.update(data.get("answers") or {})
                break
            except TransportError as exc:
                if not exc.too_long or attempt == 3 or len(kept) < 2:
                    raise
                kept = kept[: max(1, int(len(kept) * 0.85))]
        final = {k: answers[k] for k in questions if k in answers}
        missing = [k for k in questions if k not in final]
        if missing:
            raise RuntimeError(f"Jev returned no answer for: {missing}")
        sufficiency = {k: float((answers.get(SUFF + k) or {}).get("noul", 0.0)) for k in questions}
        return final, sufficiency, kept

    # ---- the drop-in call ----

    async def asystem_one(self, state: Any, questions: Mapping[str, Question]) -> LongJevResult:
        if not questions:
            raise ValueError("questions is empty")
        before = self._counters()
        raw = {k: to_json(q) for k, q in questions.items()}
        size = self.estimator.estimate(state) + sum(self.estimator.estimate(q) for q in raw.values())
        if size <= self.max_budget_tokens:
            try:
                data = await self.transport.decide(state, raw, self.model)
                answers = data.get("answers") or {}
                missing = [k for k in questions if k not in answers]
                if missing:
                    raise RuntimeError(f"Jev returned no answer for: {missing}")
                return self._result(answers, LongInputInfo(passthrough=True), before)
            except TransportError as exc:
                if not exc.too_long:
                    raise

        chunks = self.chunk(state)
        scores = await self.score(chunks, questions)
        if scores is not None and scores.ans:
            self._log_scores(state, chunks, scores)
            chunks = self.recalibrate(chunks, scores)

        budget = self.effective_budget(questions, self.budget_tokens)
        kept = self.select(chunks, scores, budget)
        answers, sufficiency, kept = await self.judge(chunks, questions, kept)
        widened = False
        ceiling = self.effective_budget(questions, self.max_budget_tokens)
        if min(sufficiency.values()) < 0.5 and budget < ceiling:
            widened = True
            kept = self.select(chunks, scores, ceiling)
            answers, sufficiency, kept = await self.judge(chunks, questions, kept)

        voted = vote(scores, questions) if scores is not None and scores.ans else {}
        info = LongInputInfo(
            passthrough=False,
            selector=self.selector,
            n_chunks=len(chunks),
            kept=[
                {
                    "id": i,
                    "label": chunks[i].label,
                    "score": None if scores is None else max(float(r[i]) for r in scores.rel.values()),
                }
                for i in sorted(kept)
            ],
            state_tokens=sum(chunks[i].est_tokens for i in kept),
            widened=widened,
            sufficiency=sufficiency,
            vote=voted,
            agreement={k: agrees(answers[k], voted.get(k)) for k in questions} if voted else {},
            failed_chunks=[] if scores is None else scores.failed,
        )
        return self._result(answers, info, before)

    def system_one(self, state: Any, questions: Mapping[str, Question]) -> LongJevResult:
        return self._run(self.asystem_one(state, questions))

    def _run(self, coro):
        """Sync calls share one event loop: the HTTP client's open connections belong to the loop
        that made them, so a fresh asyncio.run() per call breaks the second call."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            coro.close()
            raise RuntimeError(
                "system_one() cannot run inside an event loop (Jupyter, async code); "
                "use `await lj.asystem_one(...)`"
            )
        if self._loop is None or self._loop.is_closed():
            self._loop = asyncio.new_event_loop()
        return self._loop.run_until_complete(coro)

    async def aclose(self) -> None:
        close = getattr(self.transport, "aclose", None)
        if close:
            await close()

    def close(self) -> None:
        """Sync counterpart of aclose(), for code that used system_one()."""
        self._run(self.aclose())
        self._loop.close()

    # ---- helpers ----

    def _counters(self) -> tuple[int, int, int, float]:
        usage = getattr(self.transport, "usage", None)
        if usage is None:
            return (0, 0, 0, 0.0)
        return (usage.jev_input_tokens, usage.embed_tokens, usage.jev_requests + usage.embed_requests, usage.usd)

    def _result(self, answers: Mapping[str, dict], info: LongInputInfo, before) -> LongJevResult:
        after = self._counters()
        info.n_requests = after[2] - before[2]
        info.est_cost_usd = after[3] - before[3]
        usage = {"input_tokens": after[0] - before[0], "embed_tokens": after[1] - before[1]}
        return LongJevResult(
            model=self.model,
            answers={k: Answer(v) for k, v in answers.items() if not k.startswith(SUFF)},
            usage=usage,
            long_input=info,
        )

    def _log_scores(self, state: Any, chunks: list[Chunk], scores: ChunkScores, doc_id: str | None = None) -> None:
        if not self.score_log:
            return
        if doc_id is None:
            blob = state if isinstance(state, str) else json.dumps(state, sort_keys=True)
            doc_id = hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]
        self.score_log.parent.mkdir(parents=True, exist_ok=True)
        with self.score_log.open("a", encoding="utf-8") as handle:
            for chunk in chunks:
                if scores.scored is not None and not scores.scored[chunk.id]:
                    continue
                row = {
                    "doc": doc_id, "chunk": chunk.id, "label": chunk.label,
                    "start": chunk.start, "end": chunk.end, "tokens": chunk.est_tokens,
                    "scores": {
                        k: {
                            "rel": round(float(scores.rel[k][chunk.id]), 4),
                            "suff": round(float(scores.suff[k][chunk.id]), 4),
                            "ans": scores.ans[k][chunk.id],
                        }
                        for k in scores.rel
                    },
                }
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
