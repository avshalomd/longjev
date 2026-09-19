"""How requests reach Jev and the embedding model. Nothing else knows the endpoints."""

from __future__ import annotations

import asyncio
import os
import random
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx
import numpy as np

from .cache import Cache, request_key

JEV_USD_PER_MTOK = 0.042
EMBED_USD_PER_MTOK = 0.01
RETRY_STATUS = {429, 500, 502, 503, 504, 520, 521, 522, 523, 524, 529}


class TransportError(Exception):
    def __init__(self, status: int, body: str):
        super().__init__(f"HTTP {status}: {body[:300]}")
        self.status = status
        self.body = body

    @property
    def too_long(self) -> bool:
        text = self.body.lower()
        return self.status in (400, 413, 422) and any(
            w in text for w in ("max_tokens_exceeded", "too long", "token limit", "exceed")
        )


class Transport(Protocol):
    async def decide(self, state: Any, questions: dict, model: str) -> dict: ...

    async def embed(self, texts: list[str], model: str) -> list[np.ndarray]: ...


@dataclass
class Usage:
    jev_input_tokens: int = 0
    embed_tokens: int = 0
    jev_requests: int = 0
    embed_requests: int = 0
    cache_hits: int = 0
    extra: dict = field(default_factory=dict)

    @property
    def usd(self) -> float:
        return (
            self.jev_input_tokens * JEV_USD_PER_MTOK + self.embed_tokens * EMBED_USD_PER_MTOK
        ) / 1e6


def _error_code(error: Any) -> int:
    """The status inside a 200 response's error field, whatever shape the provider gave it."""
    code = error.get("code") if isinstance(error, dict) else None
    try:
        return int(code)
    except (TypeError, ValueError):  # no code, or a name instead of a number
        return 400 if "max_tokens_exceeded" in str(error) else 500


class _HTTPTransport:
    """Retries, errors and usage counting shared by the providers below."""

    key_var = ""
    default_base_url = ""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 90.0,
        max_attempts: int = 6,
        client: httpx.AsyncClient | None = None,
        sleep=asyncio.sleep,
    ):
        self.api_key = api_key or os.environ.get(self.key_var)
        if not self.api_key:
            raise RuntimeError(f"{self.key_var} is not set (export it, or pass api_key=)")
        self.base_url = (base_url or self.default_base_url).rstrip("/")
        self.max_attempts = max_attempts
        self._client = client or httpx.AsyncClient(
            timeout=timeout, limits=httpx.Limits(max_connections=128, max_keepalive_connections=20)
        )
        self._sleep = sleep
        self.usage = Usage()

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.api_key}"}

    def cache_model(self, kind: str, model: str) -> str:
        """The model name a cached response is stored under. Providers serving the same pinned
        model share entries; one that may serve a different model overrides this."""
        return model

    async def _post(self, path: str, body: dict, headers: dict | None = None) -> dict:
        headers = {**self._headers(), **(headers or {})}
        last: Exception | None = None
        for attempt in range(self.max_attempts):
            try:
                response = await self._client.post(self.base_url + path, json=body, headers=headers)
            except httpx.TransportError as exc:
                last = exc
            else:
                if response.status_code == 200:
                    try:
                        data = response.json()
                    except ValueError:  # an HTML error page with status 200
                        data = {"error": {"code": 502, "message": response.text[:200]}}
                    if isinstance(data, dict) and "error" in data and "answers" not in data and "data" not in data:
                        last = TransportError(_error_code(data["error"]), str(data["error"]))
                        if last.status not in RETRY_STATUS:
                            raise last
                    else:
                        return data
                else:
                    last = TransportError(response.status_code, response.text)
                    if response.status_code not in RETRY_STATUS:
                        raise last
            if attempt < self.max_attempts - 1:
                await self._sleep(min(30.0, 0.5 * 2**attempt) * (0.5 + random.random()))
        raise last if last else RuntimeError("request failed")

    def _count_decision(self, data: dict) -> dict:
        self.usage.jev_requests += 1
        self.usage.jev_input_tokens += int((data.get("usage") or {}).get("input_tokens", 0) or 0)
        return data

    async def _embed(self, path: str, texts: list[str], model: str) -> list[np.ndarray]:
        data = await self._post(path, {"model": model, "input": texts})
        self.usage.embed_requests += 1
        usage = data.get("usage") or {}
        self.usage.embed_tokens += int(usage.get("prompt_tokens") or usage.get("total_tokens") or 0)
        rows = sorted(data["data"], key=lambda r: r.get("index", 0))
        return [np.asarray(r["embedding"], dtype=np.float32) for r in rows]

    async def aclose(self) -> None:
        await self._client.aclose()


class OpenRouterTransport(_HTTPTransport):
    key_var = "OPENROUTER_API_KEY"
    default_base_url = "https://openrouter.ai/api"

    def _headers(self) -> dict:
        return {**super()._headers(), "X-Title": "longjev"}

    async def decide(self, state: Any, questions: dict, model: str) -> dict:
        body = {"model": model, "state": state, "questions": questions}
        return self._count_decision(await self._post("/alpha/decisions", body))

    async def embed(self, texts: list[str], model: str) -> list[np.ndarray]:
        return await self._embed("/v1/embeddings", texts, model)


# longjev names models the OpenRouter way (`typesafe/jev-1.13`, the only Jev name OpenRouter accepts).
# The other providers call the same models by these names; `typesafe/jev-latest` asks them for their
# newest Jev. A name not listed is sent unchanged.
TYPESAFE_MODELS = {"typesafe/jev-latest": "jev-latest", "typesafe/jev-1.13": "jev-1.13.0"}
# Vercel lists Jev only as `typesafe-ai/jev`, so a pinned version there runs whatever Vercel serves.
VERCEL_MODELS = {
    "typesafe/jev-latest": "typesafe-ai/jev",
    "typesafe/jev-1.13": "typesafe-ai/jev",
    "qwen/qwen3-embedding-8b": "alibaba/qwen3-embedding-8b",
}


class TypeSafeTransport(_HTTPTransport):
    """TypeSafe's own API. It has no embeddings, so `embed` goes to `embedder`, or else to
    OpenRouter or Vercel, whichever key is set first."""

    key_var = "TYPESAFE_API_KEY"
    default_base_url = "https://api.typesafe.ai"

    def __init__(self, *args, embedder: _HTTPTransport | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        if embedder is None:
            name = next((n for n in ("openrouter", "vercel") if os.environ.get(PROVIDERS[n].key_var)), None)
            shared = {k: kwargs[k] for k in ("timeout", "max_attempts", "sleep") if k in kwargs}
            embedder = PROVIDERS[name](**shared) if name else None
        self._embedder = embedder
        if embedder is not None:
            embedder.usage = self.usage  # one bill for both

    async def decide(self, state: Any, questions: dict, model: str) -> dict:
        body = {"model": TYPESAFE_MODELS.get(model, model), "state": state, "questions": questions}
        return self._count_decision(await self._post("/v1/systemone", body))

    async def embed(self, texts: list[str], model: str) -> list[np.ndarray]:
        if self._embedder is None:
            raise RuntimeError(
                "TypeSafe's API has no embeddings: the embed selectors need OPENROUTER_API_KEY or AI_GATEWAY_API_KEY"
            )
        return await self._embedder.embed(texts, model)

    def cache_model(self, kind: str, model: str) -> str:
        if kind == "embed" and self._embedder is not None:
            return self._embedder.cache_model(kind, model)
        return model

    async def aclose(self) -> None:
        await super().aclose()
        if self._embedder is not None:
            await self._embedder.aclose()


class VercelTransport(_HTTPTransport):
    """Vercel AI Gateway. Vercel documents Jev through its TypeScript AI SDK only; `decide` makes
    the HTTP call that SDK makes. Answers are converted to TypeSafe's format, so the rest of
    longjev sees the same shape from every provider."""

    key_var = "AI_GATEWAY_API_KEY"
    default_base_url = "https://ai-gateway.vercel.sh"

    async def decide(self, state: Any, questions: dict, model: str) -> dict:
        headers = {
            "ai-gateway-protocol-version": "0.0.1",
            "ai-gateway-auth-method": "api-key",
            "ai-evaluation-model-specification-version": "4",
            "ai-model-id": VERCEL_MODELS.get(model, model),
        }
        asked = {k: {**q, "type": "boolean"} if q.get("type") == "noul" else q for k, q in questions.items()}
        data = await self._post("/v4/ai/evaluation-model", {"state": state, "questions": asked}, headers)
        return self._count_decision(_from_gateway(data))

    async def embed(self, texts: list[str], model: str) -> list[np.ndarray]:
        return await self._embed("/v1/embeddings", texts, VERCEL_MODELS.get(model, model))

    def cache_model(self, kind: str, model: str) -> str:
        # `typesafe/jev-1.13` runs as the unversioned `typesafe-ai/jev` here, so its answers
        # must not be replayed as the pinned model's.
        return "vercel:" + VERCEL_MODELS.get(model, model)


def _from_gateway(data: dict) -> dict:
    """Vercel's answer format to TypeSafe's: `boolean` becomes `noul`, and Choice and Score
    confidence moves from provider metadata onto the answer."""
    confidence = ((data.get("providerMetadata") or {}).get("typesafe") or {}).get("confidence") or {}
    answers = {}
    for key, answer in (data.get("answers") or {}).items():
        if not isinstance(answer, dict):
            raise TransportError(502, f"unexpected answer for {key!r}: {answer!r}")
        if answer.get("type") == "boolean":
            probability = answer.get("probability")
            if not isinstance(probability, (int, float)):
                raise TransportError(502, f"unexpected yes/no answer for {key!r}: fields {sorted(answer)}")
            answer = {"type": "noul", "noul": float(probability)}
        elif key in confidence and "confidence" not in answer:
            answer = {**answer, "confidence": confidence[key]}
        answers[key] = answer
    usage = data.get("usage") or {}
    out = {"answers": answers, "usage": {"input_tokens": usage.get("inputTokens", usage.get("input_tokens", 0)),
                                         "output_tokens": usage.get("outputTokens", usage.get("output_tokens", 0))}}
    if data.get("model"):
        out["model"] = data["model"]
    return out


PROVIDERS = {"typesafe": TypeSafeTransport, "openrouter": OpenRouterTransport, "vercel": VercelTransport}


def make_transport(provider: str | None = None, **kwargs) -> Transport:
    """The transport for `provider`, else the LONGJEV_PROVIDER variable, else the first provider
    (in PROVIDERS order) whose key is set."""
    name = provider or os.environ.get("LONGJEV_PROVIDER", "").strip()
    if not name:
        name = next((n for n, cls in PROVIDERS.items() if os.environ.get(cls.key_var)), "")
        if not name:
            keys = ", ".join(cls.key_var for cls in PROVIDERS.values())
            raise RuntimeError(f"No API key is set: export one of {keys}")
    if name not in PROVIDERS:
        raise ValueError(f"provider must be one of {tuple(PROVIDERS)}, not {name!r}")
    return PROVIDERS[name](**kwargs)


class CachedTransport:
    """Wraps a transport with a disk cache. Usage counts only real requests."""

    def __init__(self, inner: Transport, cache: Cache):
        self.inner = inner
        self.cache = cache
        self._model = getattr(inner, "cache_model", lambda kind, model: model)

    @property
    def usage(self) -> Usage:
        return self.inner.usage  # type: ignore[attr-defined]

    async def decide(self, state: Any, questions: dict, model: str) -> dict:
        key = request_key("decide", self._model("decide", model), state, questions)
        hit = self.cache.get_response(key)
        if hit is not None:
            self.usage.cache_hits += 1
            return hit
        data = await self.inner.decide(state, questions, model)
        self.cache.put_response(key, data)
        return data

    async def embed(self, texts: list[str], model: str) -> list[np.ndarray]:
        keys = [request_key("embed", self._model("embed", model), t) for t in texts]
        found = [self.cache.get_vector(k) for k in keys]
        missing = [i for i, v in enumerate(found) if v is None]
        if missing:
            fresh = await self.inner.embed([texts[i] for i in missing], model)
            for i, vec in zip(missing, fresh):
                self.cache.put_vector(keys[i], vec)
                found[i] = vec
        return found  # type: ignore[return-value]

    async def aclose(self) -> None:
        self.cache.flush()
        close = getattr(self.inner, "aclose", None)
        if close:
            await close()
