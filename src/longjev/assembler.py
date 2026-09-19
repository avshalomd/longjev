"""Builds the reduced state: pick chunks within a budget, restore order, mark gaps."""

from __future__ import annotations

from typing import Mapping

import numpy as np

from .chunker import Chunk

GAP_TOKENS = 15  # rough cost of one "[… omitted …]" line


def select_round_robin(
    chunks: list[Chunk], rel: Mapping[str, np.ndarray], budget_tokens: int
) -> list[int]:
    """Each question in turn takes its best unselected chunk until the budget is full.
    Returns chunk ids in selection order (best first)."""
    orders = {k: list(np.argsort(-scores, kind="stable")) for k, scores in rel.items()}
    position = {k: 0 for k in orders}
    chosen: list[int] = []
    taken: set[int] = set()
    used = 0
    active = list(orders)
    while active:
        for key in list(active):
            order = orders[key]
            while position[key] < len(order) and int(order[position[key]]) in taken:
                position[key] += 1
            if position[key] >= len(order):
                active.remove(key)
                continue
            cid = int(order[position[key]])
            cost = chunks[cid].est_tokens + GAP_TOKENS
            if used + cost > budget_tokens:
                active.remove(key)
                continue
            chosen.append(cid)
            taken.add(cid)
            used += cost
    if not chosen and orders:  # no piece fits the budget: keep the single best one
        first = next(iter(orders.values()))
        if len(first):
            chosen.append(int(first[0]))
    return chosen


def select_truncate(chunks: list[Chunk], budget_tokens: int) -> list[int]:
    """The floor: keep the start and the end, drop the middle."""
    half = budget_tokens // 2
    head: list[int] = []
    used = 0
    for chunk in chunks:
        if used + chunk.est_tokens > half:
            break
        head.append(chunk.id)
        used += chunk.est_tokens
    tail: list[int] = []
    used = 0
    for chunk in reversed(chunks[len(head) :]):
        if used + chunk.est_tokens > half - GAP_TOKENS:
            break
        tail.append(chunk.id)
        used += chunk.est_tokens
    if not head and not tail and chunks:  # no piece fits the budget: keep the opening
        return [chunks[0].id]
    return head + tail[::-1]


def _path(label: str) -> str:
    return label.split("#c")[0]


def render_state(chunks: list[Chunk], kept: list[int]) -> str:
    """Kept chunks in original order, with a line wherever text was left out."""
    ids = sorted(set(kept))
    parts: list[str] = []
    previous = -1
    for cid in ids:
        if cid != previous + 1:
            omitted = sum(c.est_tokens for c in chunks[previous + 1 : cid])
            parts.append(f"\n[… about {omitted} tokens omitted …]\n")
        elif parts and _path(chunks[cid].label) != _path(chunks[previous].label):
            parts.append("\n")
        parts.append(chunks[cid].text)
        previous = cid
    if ids and ids[-1] != len(chunks) - 1:
        omitted = sum(c.est_tokens for c in chunks[ids[-1] + 1 :])
        parts.append(f"\n[… about {omitted} tokens omitted …]\n")
    return "".join(parts)
