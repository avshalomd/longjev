import json
import os

import httpx
import pytest

from longjev import (
    Cache, CachedTransport, Choice, LongJev, Noul, OpenRouterTransport, TokenEstimator, TransportError, TypeSafeTransport,
    VercelTransport, make_transport,
)
from tests.fakes import FakeTransport

QUESTIONS = {"q": Choice("Which letter is the secret?", ["A", "B", "C", "D"])}


def haystack(n=300, needle_at=170):
    paragraphs = [f"Paragraph {i}. " + "filler words here " * 40 for i in range(n)]
    paragraphs[needle_at] += " SECRET: the letter is C."
    return "\n\n".join(paragraphs)


def make(selector="jev", transport=None, **kw):
    return LongJev(
        selector=selector, transport=transport or FakeTransport(),
        estimator=TokenEstimator(4.0), budget_tokens=4_000, **kw,
    )


async def test_passthrough_when_it_fits():
    fake = FakeTransport()
    result = await make(transport=fake).asystem_one("short SECRET text", QUESTIONS)
    assert result.long_input.passthrough and len(fake.decide_calls) == 1
    assert result.answers["q"].choice == "C"


@pytest.mark.parametrize("selector", ["jev", "embed", "embed_then_jev"])
async def test_selectors_find_the_needle(selector):
    fake = FakeTransport()
    result = await make(selector, fake).asystem_one(haystack(), QUESTIONS)
    info = result.long_input
    assert result.answers["q"].choice == "C"
    assert not info.passthrough and info.selector == selector
    assert info.state_tokens <= 4_000 * 1.1
    assert "suff__q" not in result.answers and info.sufficiency["q"] > 0.5
    if selector != "embed":
        assert info.vote["q"]["choice"] == "C" and info.agreement["q"] is True
    if selector == "embed_then_jev":
        scored = [c for c in fake.decide_calls if "rel__q" in c[1]]
        assert len(scored) < info.n_chunks  # Jev read only the shortlist


async def test_truncate_misses_a_needle_in_the_middle_and_widens():
    fake = FakeTransport()
    result = await make("truncate", fake).asystem_one(haystack(), QUESTIONS)
    assert result.answers["q"].choice == "A"
    assert result.long_input.widened and result.long_input.vote == {}


async def test_too_long_drops_tail_and_retries():
    fake = FakeTransport(max_state_chars=13_000)
    result = await make("jev", fake).asystem_one(haystack(), QUESTIONS)
    assert result.answers["q"].choice == "C"


async def test_failed_chunks_are_reported_and_many_failures_raise():
    text = haystack()
    ok = await make("jev", FakeTransport(fail_on="Paragraph 7.")).asystem_one(text, QUESTIONS)
    assert ok.long_input.failed_chunks
    with pytest.raises(RuntimeError):
        await make("jev", FakeTransport(fail_on="filler")).asystem_one(text, QUESTIONS)


async def test_cache_avoids_repeat_requests_and_score_log_is_written(tmp_path):
    fake = FakeTransport()
    cache = Cache(tmp_path / "c.sqlite")
    log = tmp_path / "scores.jsonl"
    lj = make("jev", fake, cache=cache, score_log=log)
    await lj.asystem_one(haystack(), QUESTIONS)
    first = len(fake.decide_calls)
    await lj.asystem_one(haystack(), QUESTIONS)
    assert len(fake.decide_calls) == first
    row = json.loads(log.read_text().splitlines()[0])
    assert {"doc", "chunk", "label", "scores"} <= set(row) and "rel" in row["scores"]["q"]


async def test_multiple_question_types():
    questions = {"n": Noul("Is there a secret?"), "q": QUESTIONS["q"]}
    result = await make("jev").asystem_one(haystack(), questions)
    assert result.answers["n"].noul > 0.5 and result.answers["q"].choice == "C"


def _transport(handler, cls=OpenRouterTransport):
    async def no_sleep(_):
        return None
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return cls(api_key="test", client=client, sleep=no_sleep)


async def test_retries_on_429_then_succeeds():
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) < 3:
            return httpx.Response(429, text="slow down")
        return httpx.Response(200, json={"answers": {"q": {"type": "noul", "noul": 1}}, "usage": {"input_tokens": 7}})

    transport = _transport(handler)
    data = await transport.decide("s", {"q": {"type": "noul", "instructions": "x"}}, "m")
    assert len(calls) == 3 and data["answers"]["q"]["noul"] == 1
    assert transport.usage.jev_input_tokens == 7
    body = json.loads(calls[0].content)
    assert body["model"] == "m" and calls[0].url.path == "/api/alpha/decisions"


async def test_401_raises_immediately():
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(401, text="bad key")

    with pytest.raises(TransportError):
        await _transport(handler).decide("s", {}, "m")
    assert len(calls) == 1


def test_missing_key_is_a_clear_error(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        OpenRouterTransport()


LIVE = [
    pytest.param(name, marks=pytest.mark.skipif(not os.environ.get(cls.key_var), reason=f"needs {cls.key_var}"))
    for name, cls in {"typesafe": TypeSafeTransport, "openrouter": OpenRouterTransport, "vercel": VercelTransport}.items()
]


@pytest.mark.parametrize("provider", LIVE)
async def test_live_smoke(provider):
    lj = LongJev(provider=provider)
    result = await lj.asystem_one(
        "Help! My payouts have been failing for 3 days.",
        {"urgent": Noul("Does this convey urgency?"), "team": Choice("Which team?", ["payments", "sales", "other"])},
    )
    assert result.answers["urgent"].noul > 0.5 and result.answers["team"].choice == "payments"
    await lj.aclose()


@pytest.mark.parametrize("provider", LIVE)
async def test_live_embeddings(provider):
    if provider == "typesafe" and not (os.environ.get("OPENROUTER_API_KEY") or os.environ.get("AI_GATEWAY_API_KEY")):
        pytest.skip("TypeSafe embeds through OpenRouter or Vercel; neither key is set")
    transport = make_transport(provider)
    vectors = await transport.embed(["payouts are failing"], "qwen/qwen3-embedding-8b")
    assert len(vectors) == 1 and vectors[0].size > 100
    await transport.aclose()


def test_system_one_can_be_called_repeatedly_and_closed():
    lj = make(transport=FakeTransport())
    for _ in range(3):
        assert lj.system_one("short SECRET text", QUESTIONS).answers["q"].choice == "C"
    lj.close()


async def test_system_one_inside_a_running_loop_points_to_the_async_call():
    with pytest.raises(RuntimeError, match="asystem_one"):
        make().system_one("short text", QUESTIONS)


async def test_passthrough_with_a_missing_answer_raises():
    class Partial(FakeTransport):
        async def decide(self, state, questions, model):
            return {"answers": {}}

    with pytest.raises(RuntimeError, match="no answer"):
        await make(transport=Partial()).asystem_one("short text", QUESTIONS)


async def test_odd_error_bodies_become_transport_errors():
    bodies = [
        httpx.Response(200, json={"error": "max_tokens_exceeded: state too long"}),
        httpx.Response(200, json={"error": {"code": "bad_request", "message": "nope"}}),
    ]
    for body in bodies:
        with pytest.raises(TransportError):
            await _transport(lambda request, body=body: body).decide("s", {}, "m")
    with pytest.raises(TransportError) as caught:
        await _transport(lambda request: bodies[0]).decide("s", {}, "m")
    assert caught.value.too_long


async def test_embed_selector_with_nothing_to_embed():
    from longjev.selectors import score_embed

    scores = await score_embed([], QUESTIONS, FakeTransport(), "m")
    assert scores.rel["q"].shape == (0,)


NOUL_AND_CHOICE = {
    "n": {"type": "noul", "instructions": "x"},
    "c": {"type": "choice", "instructions": "y", "criteria": {"a": "A", "b": "B"}},
}


async def test_typesafe_transport_calls_the_native_api(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("AI_GATEWAY_API_KEY", raising=False)
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, json={"model": "jev-1.13.0", "answers": {"n": {"type": "noul", "noul": 0.9}},
                                         "usage": {"input_tokens": 11}})

    transport = _transport(handler, TypeSafeTransport)
    data = await transport.decide("s", {"n": NOUL_AND_CHOICE["n"]}, "typesafe/jev-1.13")
    body = json.loads(calls[0].content)
    assert str(calls[0].url) == "https://api.typesafe.ai/v1/systemone"
    assert calls[0].headers["authorization"] == "Bearer test"
    assert body["model"] == "jev-1.13.0" and body["questions"]["n"]["type"] == "noul"
    assert data["answers"]["n"]["noul"] == 0.9 and transport.usage.jev_input_tokens == 11
    with pytest.raises(RuntimeError, match="no embeddings"):
        await transport.embed(["x"], "qwen/qwen3-embedding-8b")


async def test_typesafe_embeddings_go_to_the_embedder_on_the_same_bill():
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.0, 0.0]}], "usage": {"prompt_tokens": 5}})

    embedder = _transport(handler)  # an OpenRouter transport on a fake network
    transport = TypeSafeTransport(api_key="test", embedder=embedder)
    vectors = await transport.embed(["x"], "qwen/qwen3-embedding-8b")
    assert len(vectors) == 1 and calls[0].url.path == "/api/v1/embeddings"
    assert transport.usage.embed_tokens == 5 and transport.usage.embed_requests == 1
    await transport.aclose()


async def test_vercel_transport_speaks_typesafe_format_to_the_rest_of_longjev():
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, json={
            "answers": {"n": {"type": "boolean", "probability": 0.2},
                        "c": {"type": "choice", "choice": "b", "probabilities": {"a": 0.3, "b": 0.7}}},
            "providerMetadata": {"typesafe": {"confidence": {"c": 0.4}}},
            "usage": {"inputTokens": 13, "outputTokens": 2},
        })

    transport = _transport(handler, VercelTransport)
    data = await transport.decide("s", NOUL_AND_CHOICE, "typesafe/jev-latest")
    body = json.loads(calls[0].content)
    assert str(calls[0].url) == "https://ai-gateway.vercel.sh/v4/ai/evaluation-model"
    assert calls[0].headers["ai-model-id"] == "typesafe-ai/jev" and "model" not in body
    assert body["questions"]["n"]["type"] == "boolean" and NOUL_AND_CHOICE["n"]["type"] == "noul"
    assert data["answers"]["n"] == {"type": "noul", "noul": 0.2}
    assert data["answers"]["c"]["confidence"] == 0.4 and data["answers"]["c"]["choice"] == "b"
    assert transport.usage.jev_input_tokens == 13


async def test_vercel_embeddings_use_the_gateway_model_name():
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.0, 0.0]}], "usage": {"prompt_tokens": 3}})

    transport = _transport(handler, VercelTransport)
    vectors = await transport.embed(["x"], "qwen/qwen3-embedding-8b")
    assert json.loads(calls[0].content)["model"] == "alibaba/qwen3-embedding-8b"
    assert calls[0].url.path == "/v1/embeddings" and len(vectors) == 1 and transport.usage.embed_tokens == 3


async def test_vercel_answer_in_an_unknown_shape_is_an_error():
    def handler(request):
        return httpx.Response(200, json={"answers": {"n": {"type": "boolean", "value": True}}})

    with pytest.raises(TransportError, match="fields"):
        await _transport(handler, VercelTransport).decide("s", {"n": NOUL_AND_CHOICE["n"]}, "typesafe/jev-latest")


async def test_cache_keeps_vercel_answers_apart_from_the_pinned_model(tmp_path):
    def handler(request):
        if "evaluation-model" in request.url.path:
            return httpx.Response(200, json={"answers": {"n": {"type": "boolean", "probability": 0.9}}})
        return httpx.Response(200, json={"answers": {"n": {"type": "noul", "noul": 0.1}}})

    cache = Cache(tmp_path / "c.sqlite")
    vercel = CachedTransport(_transport(handler, VercelTransport), cache)
    openrouter = CachedTransport(_transport(handler), cache)
    q = {"n": {"type": "noul", "instructions": "?"}}
    assert (await vercel.decide("s", q, "typesafe/jev-1.13"))["answers"]["n"]["noul"] == 0.9
    assert (await openrouter.decide("s", q, "typesafe/jev-1.13"))["answers"]["n"]["noul"] == 0.1
    assert openrouter.usage.jev_requests == 1
    await vercel.aclose()
    await openrouter.aclose()


async def test_typesafe_passes_its_retry_settings_to_the_embedder(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")

    async def no_sleep(_):
        return None

    transport = TypeSafeTransport(api_key="test", max_attempts=2, sleep=no_sleep)
    assert transport._embedder.max_attempts == 2 and transport._embedder._sleep is no_sleep
    await transport.aclose()


def test_provider_choice(monkeypatch):
    for var in ("OPENROUTER_API_KEY", "TYPESAFE_API_KEY", "AI_GATEWAY_API_KEY", "LONGJEV_PROVIDER"):
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(RuntimeError, match="TYPESAFE_API_KEY"):
        make_transport()
    monkeypatch.setenv("AI_GATEWAY_API_KEY", "v")
    assert isinstance(make_transport(), VercelTransport)
    monkeypatch.setenv("OPENROUTER_API_KEY", "o")
    assert isinstance(make_transport(), OpenRouterTransport)
    monkeypatch.setenv("TYPESAFE_API_KEY", "t")
    assert isinstance(make_transport(), TypeSafeTransport)  # TypeSafe first whenever its key is set
    monkeypatch.setenv("LONGJEV_PROVIDER", "vercel")
    assert isinstance(make_transport(), VercelTransport)
    assert isinstance(make_transport("typesafe"), TypeSafeTransport)
    with pytest.raises(ValueError):
        make_transport("anthropic")
