"""Token estimates. Jev's tokenizer is not public, so size is estimated from characters."""

from __future__ import annotations

import json
import math
from typing import Any


class TokenEstimator:
    """Estimates tokens as characters / chars_per_token.

    The ratio is fixed by default so chunk boundaries and cache keys stay stable
    between runs. `observe` records real usage so drift can be reported; with
    `adaptive=True` it also updates the ratio.
    """

    def __init__(self, chars_per_token: float = 3.0, adaptive: bool = False):
        self.chars_per_token = chars_per_token
        self.adaptive = adaptive
        self._chars = 0
        self._tokens = 0

    def estimate(self, value: Any) -> int:
        text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        return math.ceil(len(text) / self.chars_per_token)

    def chars_for(self, tokens: int) -> int:
        return int(tokens * self.chars_per_token)

    def observe(self, chars: int, tokens: int) -> None:
        if chars <= 0 or tokens <= 0:
            return
        self._chars += chars
        self._tokens += tokens
        if self.adaptive:
            self.chars_per_token = self._chars / self._tokens

    @property
    def observed_chars_per_token(self) -> float | None:
        return self._chars / self._tokens if self._tokens else None
