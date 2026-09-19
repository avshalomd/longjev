"""Combines per-chunk answers into a vote and compares it with the final judgment."""

from __future__ import annotations

from typing import Mapping

import numpy as np

from .questions import Question, to_json
from .selectors import ChunkScores

MIN_WEIGHT = 1e-6


def vote(scores: ChunkScores, task_questions: Mapping[str, Question]) -> dict[str, dict | None]:
    """Each chunk's answer, weighted by relevance × sufficiency."""
    out: dict[str, dict | None] = {}
    for key, question in task_questions.items():
        kind = to_json(question)["type"]
        answers = scores.ans.get(key) or []
        weights = scores.rel[key] * scores.suff[key] if key in scores.suff else None
        if weights is None:
            out[key] = None
            continue
        total = 0.0
        noul_sum = 0.0
        dist: dict[str, float] = {}
        for i, answer in enumerate(answers):
            w = float(weights[i])
            if not answer or w <= 0:
                continue
            if kind != "noul" and not answer.get("probabilities"):
                continue
            total += w
            if kind == "noul":
                noul_sum += w * float(answer.get("noul", 0.0))
            else:
                for option, p in (answer.get("probabilities") or {}).items():
                    dist[option] = dist.get(option, 0.0) + w * float(p)
        if total < MIN_WEIGHT:
            out[key] = None
        elif kind == "noul":
            out[key] = {"type": "noul", "noul": noul_sum / total}
        else:
            probs = {o: v / total for o, v in dist.items()}
            top = max(probs, key=probs.get)  # type: ignore[arg-type]
            if kind == "choice":
                out[key] = {"type": "choice", "choice": top, "probabilities": probs}
            else:
                try:
                    expected = sum(float(level) * p for level, p in probs.items())
                except ValueError:  # levels keyed by name, not number: report the likeliest level only
                    out[key] = {"type": "score", "score": top, "probabilities": probs}
                else:
                    out[key] = {"type": "score", "score": expected, "probabilities": probs}
    return out


def agrees(final: dict | None, voted: dict | None) -> bool | None:
    if not final or not voted:
        return None
    kind = final.get("type")
    if kind == "choice":
        return final.get("choice") == voted.get("choice")
    if kind == "noul":
        return (final.get("noul", 0) >= 0.5) == (voted.get("noul", 0) >= 0.5)
    if kind == "score":
        return round(final.get("score", 0)) == round(voted.get("score", 0))
    return None


def top_weights(scores: ChunkScores, key: str, n: int = 5) -> list[tuple[int, float]]:
    weights = scores.rel[key] * scores.suff[key]
    order = np.argsort(-weights)[:n]
    return [(int(i), float(weights[i])) for i in order]
