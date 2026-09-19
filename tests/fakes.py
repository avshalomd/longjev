"""A fake Jev: answers from keywords so tests need no network."""

from __future__ import annotations

import numpy as np

from longjev.transport import TransportError, Usage


class FakeTransport:
    """rel/suff are high when the state contains `needle`. The task Choice answers
    `answer` when the needle is present and "A" otherwise."""

    def __init__(self, needle="SECRET", answer="C", max_state_chars=None, fail_on=None):
        self.needle = needle
        self.answer = answer
        self.max_state_chars = max_state_chars
        self.fail_on = fail_on
        self.usage = Usage()
        self.decide_calls: list[tuple[str, dict]] = []
        self.embed_calls = 0

    async def decide(self, state, questions, model):
        text = state if isinstance(state, str) else str(state)
        self.decide_calls.append((text, questions))
        if self.fail_on and self.fail_on in text:
            raise TransportError(500, "boom")
        if self.max_state_chars and len(text) > self.max_state_chars:
            raise TransportError(422, "state too long: exceeds maximum context")
        present = self.needle in text
        answers = {}
        for key, q in questions.items():
            if q["type"] == "noul":
                answers[key] = {"type": "noul", "noul": 0.95 if present else 0.03}
            elif q["type"] == "choice":
                options = list(q["criteria"])
                pick = self.answer if present and self.answer in options else options[0]
                rest = 0.1 / max(1, len(options) - 1)
                answers[key] = {
                    "type": "choice", "choice": pick, "confidence": 0.8,
                    "probabilities": {o: (0.9 if o == pick else rest) for o in options},
                }
            else:
                levels = q["criteria"]
                top = len(levels) - 1 if present else 0
                answers[key] = {
                    "type": "score", "score": float(top), "confidence": 0.8,
                    "legend": {str(i): l for i, l in enumerate(levels)},
                    "probabilities": {str(i): (1.0 if i == top else 0.0) for i in range(len(levels))},
                }
        self.usage.jev_requests += 1
        self.usage.jev_input_tokens += len(text) // 4
        return {"model": model, "answers": answers, "usage": {"input_tokens": len(text) // 4}}

    async def embed(self, texts, model):
        self.embed_calls += 1
        self.usage.embed_requests += 1
        out = []
        for t in texts:
            hit = 1.0 if (self.needle in t or "Query:" in t) else 0.0
            out.append(np.array([hit, 1.0 - hit, 0.1], dtype=np.float32))
        return out
