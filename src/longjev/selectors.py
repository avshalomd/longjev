"""Scoring: how relevant is each chunk to each task question."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Mapping

import numpy as np

from .chunker import Chunk
from .questions import ANS, REL, SUFF, Question, chunk_questions, render
from .tokens import TokenEstimator
from .transport import Transport

SELECTORS = ("truncate", "embed", "jev", "embed_then_jev")
EMBED_INSTRUCTION = "Given a question, retrieve passages that help answer it"
TOTAL_LIMIT_TOKENS = 58_000  # 64K limit on state plus all questions, with margin


@dataclass
class ChunkScores:
    rel: dict[str, np.ndarray]  # task question key -> relevance per chunk
    suff: dict[str, np.ndarray] = field(default_factory=dict)
    ans: dict[str, list[dict | None]] = field(default_factory=dict)
    scored: np.ndarray | None = None  # which chunks Jev actually read
    failed: list[int] = field(default_factory=list)
    chars_per_token: float | None = None  # measured from Jev's reported usage on this state
    tokens: np.ndarray | None = None  # measured tokens per chunk, 0 where unknown
    input_tokens: int = 0  # everything the scoring pass sent to Jev, cached or not


def split_questions(
    questions: Mapping[str, dict], state_tokens: int, estimator: TokenEstimator
) -> list[dict[str, dict]]:
    """Groups of questions so each request stays under the total-size limit."""
    groups: list[dict[str, dict]] = [{}]
    used = state_tokens
    for key, question in questions.items():
        size = estimator.estimate(question)
        if groups[-1] and used + size > TOTAL_LIMIT_TOKENS:
            groups.append({})
            used = state_tokens
        groups[-1][key] = question
        used += size
    return groups


async def score_jev(
    chunks: list[Chunk],
    task_questions: Mapping[str, Question],
    transport: Transport,
    model: str,
    estimator: TokenEstimator,
    concurrency: int = 16,
    only: list[int] | None = None,
    on_done=None,
    gate: asyncio.Semaphore | None = None,
) -> ChunkScores:
    keys = list(task_questions)
    n = len(chunks)
    rel = {k: np.zeros(n, dtype=np.float32) for k in keys}
    suff = {k: np.zeros(n, dtype=np.float32) for k in keys}
    ans: dict[str, list[dict | None]] = {k: [None] * n for k in keys}
    scored = np.zeros(n, dtype=bool)
    failed: list[int] = []
    questions = chunk_questions(task_questions)
    gate = gate or asyncio.Semaphore(concurrency)
    question_tokens = sum(len(str(q)) for q in questions.values()) / 3.5
    seen = [0, 0.0, 0]  # state characters, state tokens, all input tokens
    tokens = np.zeros(n, dtype=np.int32)
    try:  # a one-character state measures what the questions alone cost
        probe = await transport.decide("-", questions, model)
        measured_q = int((probe.get("usage") or {}).get("input_tokens") or 0) - 1
        if measured_q > 0:
            question_tokens = measured_q
    except Exception:
        pass

    async def one(chunk: Chunk) -> None:
        async with gate:
            try:
                answers: dict = {}
                for group in split_questions(questions, chunk.est_tokens, estimator):
                    data = await transport.decide(chunk.text, group, model)
                    answers.update(data.get("answers") or {})
                    used = (data.get("usage") or {}).get("input_tokens")
                    seen[2] += int(used or 0)
                    if used and len(questions) == len(group) and used > question_tokens:
                        seen[0] += len(chunk.text)
                        seen[1] += used - question_tokens
                        tokens[chunk.id] = max(1, int(used - question_tokens))
            except Exception:
                failed.append(chunk.id)
                return
            finally:
                if on_done:
                    on_done()
        for k in keys:
            rel[k][chunk.id] = float((answers.get(REL + k) or {}).get("noul", 0.0))
            suff[k][chunk.id] = float((answers.get(SUFF + k) or {}).get("noul", 0.0))
            ans[k][chunk.id] = answers.get(ANS + k)
        scored[chunk.id] = True

    targets = chunks if only is None else [chunks[i] for i in only]
    await asyncio.gather(*(one(c) for c in targets))
    measured = seen[0] / seen[1] if seen[1] > 0 else None
    if measured:
        estimator.observe(seen[0], int(seen[1]))
    return ChunkScores(
        rel=rel, suff=suff, ans=ans, scored=scored, failed=sorted(failed),
        chars_per_token=measured, tokens=tokens, input_tokens=seen[2],
    )


def _batches(texts: list[str], max_items: int = 32, max_chars: int = 120_000):
    batch: list[int] = []
    size = 0
    for i, text in enumerate(texts):
        if batch and (len(batch) >= max_items or size + len(text) > max_chars):
            yield batch
            batch, size = [], 0
        batch.append(i)
        size += len(text)
    if batch:
        yield batch


async def score_embed(
    chunks: list[Chunk],
    task_questions: Mapping[str, Question],
    transport: Transport,
    embed_model: str,
    concurrency: int = 8,
) -> ChunkScores:
    if not chunks:
        return ChunkScores(rel={k: np.zeros(0, dtype=np.float32) for k in task_questions})
    texts = [c.text for c in chunks]
    vectors: list[np.ndarray | None] = [None] * len(texts)
    gate = asyncio.Semaphore(concurrency)

    async def one(batch: list[int]) -> None:
        async with gate:
            out = await transport.embed([texts[i] for i in batch], embed_model)
        for i, vec in zip(batch, out):
            vectors[i] = vec

    await asyncio.gather(*(one(b) for b in _batches(texts)))
    matrix = np.vstack(vectors)  # type: ignore[arg-type]
    matrix = matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-9)

    keys = list(task_questions)
    queries = [f"Instruct: {EMBED_INSTRUCTION}\nQuery: {render(task_questions[k])}" for k in keys]
    qvecs = await transport.embed(queries, embed_model)
    rel = {}
    for k, q in zip(keys, qvecs):
        q = q / max(float(np.linalg.norm(q)), 1e-9)
        rel[k] = (matrix @ q).astype(np.float32)
    return ChunkScores(rel=rel)


def shortlist(embed_rel: Mapping[str, np.ndarray], fraction: float = 0.25, minimum: int = 40) -> list[int]:
    """Union over questions of the top chunks by embedding similarity."""
    picked: set[int] = set()
    for scores in embed_rel.values():
        n = len(scores)
        take = min(n, max(minimum, int(np.ceil(n * fraction))))
        picked.update(int(i) for i in np.argsort(-scores)[:take])
    return sorted(picked)


def combine_shortlist(embed: ChunkScores, jev: ChunkScores, ids: list[int]) -> ChunkScores:
    """Jev's relevance inside the shortlist; outside it, chunks rank after every
    shortlisted chunk, in embedding order."""
    mask = np.zeros(len(next(iter(embed.rel.values()))), dtype=bool)
    mask[ids] = True
    rel = {k: np.where(mask, jev.rel[k], embed.rel[k] - 2.0).astype(np.float32) for k in embed.rel}
    suff = {k: np.where(mask, jev.suff[k], 0.0).astype(np.float32) for k in jev.suff}
    ans = {k: [a if mask[i] else None for i, a in enumerate(v)] for k, v in jev.ans.items()}
    return ChunkScores(
        rel=rel, suff=suff, ans=ans,
        scored=mask & (jev.scored if jev.scored is not None else mask),
        failed=[f for f in jev.failed if mask[f]], chars_per_token=jev.chars_per_token,
        tokens=jev.tokens,
    )
