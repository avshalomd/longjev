"""Splits a state of any shape into chunks of roughly equal size."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .tokens import TokenEstimator

SEPARATORS = ("\n\n", "\n", ". ", " ")


@dataclass(frozen=True)
class Chunk:
    id: int
    label: str  # JSON path plus chunk index; version 2 attaches real structure here
    start: int  # character offsets within the value the label points at
    end: int
    text: str
    est_tokens: int


def _pieces(text: str, limit: int, seps: tuple[str, ...] = SEPARATORS) -> list[str]:
    """Pieces no longer than `limit`, cut at the coarsest separator available.
    Joining the pieces reproduces `text` exactly."""
    if len(text) <= limit:
        return [text]
    for i, sep in enumerate(seps):
        if sep in text:
            parts = text.split(sep)
            parts = [p + sep for p in parts[:-1]] + [parts[-1]]
            out: list[str] = []
            for part in parts:
                if part:
                    out.extend(_pieces(part, limit, seps[i + 1 :]))
            return out
    return [text[i : i + limit] for i in range(0, len(text), limit)]


def _pack(pieces: list[str], limit: int) -> list[str]:
    out: list[str] = []
    current: list[str] = []
    size = 0
    for piece in pieces:
        if current and size + len(piece) > limit:
            out.append("".join(current))
            current, size = [], 0
        current.append(piece)
        size += len(piece)
    if current:
        out.append("".join(current))
    return out


def _chunk_text(text: str, path: str, limit: int) -> list[tuple[str, int, int, str]]:
    out = []
    offset = 0
    for i, body in enumerate(_pack(_pieces(text, limit), limit)):
        out.append((f"{path}#c{i}", offset, offset + len(body), body))
        offset += len(body)
    return out


def _chunk_list(items: list, path: str, limit: int) -> list[tuple[str, int, int, str]]:
    out = []
    texts = [i if isinstance(i, str) else json.dumps(i, ensure_ascii=False) for i in items]
    group: list[str] = []
    first = 0
    size = 0

    def flush(upto: int) -> None:
        nonlocal group, size, first
        if group:
            body = "\n".join(group)
            out.append((f"{path}[{first}:{upto}]", 0, len(body), body))
        group, size, first = [], 0, upto

    for index, text in enumerate(texts):
        if len(text) > limit:
            flush(index)
            out.extend(_chunk_text(text, f"{path}[{index}]", limit))
            first = index + 1
            continue
        if group and size + len(text) > limit:
            flush(index)
        group.append(text)
        size += len(text) + 1
    flush(len(texts))
    return out


def _chunk_value(value: Any, path: str, limit: int) -> list[tuple[str, int, int, str]]:
    if isinstance(value, str):
        return _chunk_text(value, path, limit)
    if isinstance(value, list):
        return _chunk_list(value, path, limit)
    if isinstance(value, dict):
        out = []
        for key, inner in value.items():
            pieces = _chunk_value(inner, f"{path}.{key}", limit)
            if pieces:  # the first piece of each field carries the field name
                label, start, end, text = pieces[0]
                pieces[0] = (label, start, end, f"{key}: {text}")
            out.extend(pieces)
        return out
    return _chunk_text(json.dumps(value, ensure_ascii=False), path, limit)


def chunk_state(state: Any, chunk_tokens: int, estimator: TokenEstimator) -> list[Chunk]:
    limit = max(1, estimator.chars_for(chunk_tokens))
    raw = [r for r in _chunk_value(state, "$", limit) if r[3]]
    return [
        Chunk(i, label, start, end, text, estimator.estimate(text))
        for i, (label, start, end, text) in enumerate(raw)
    ]
