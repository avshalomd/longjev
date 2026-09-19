import numpy as np
import pytest

from longjev.assembler import render_state, select_round_robin, select_truncate
from longjev.cache import Cache, request_key
from longjev.chunker import chunk_state
from longjev.questions import ANS, REL, SUFF, Choice, Noul, Score, chunk_questions, final_questions, render, to_json
from longjev.reducer import agrees, vote
from longjev.selectors import ChunkScores, shortlist, split_questions
from longjev.tokens import TokenEstimator

EST = TokenEstimator(chars_per_token=4.0)


def test_question_json_shapes():
    assert Noul("Urgent?").to_json() == {"type": "noul", "instructions": "Urgent?"}
    assert Choice("Team?", ["a", "b"]).to_json()["criteria"] == {"a": None, "b": None}
    assert Score("Angry?", ["calm", "angry"]).to_json()["criteria"] == ["calm", "angry"]
    assert to_json({"type": "noul", "instructions": "x"})["type"] == "noul"
    with pytest.raises(ValueError):
        to_json({"type": "essay", "instructions": "x"})


def test_render_includes_options():
    text = render(Choice("Which law?", {"NY": "New York", "DE": None}))
    assert "Which law?" in text and "NY: New York" in text and "DE" in text


def test_chunk_and_final_questions():
    qs = chunk_questions({"q": Choice("Which?", ["A", "B"])})
    assert set(qs) == {REL + "q", ANS + "q", SUFF + "q"}
    assert qs[ANS + "q"]["type"] == "choice" and qs[REL + "q"]["type"] == "noul"
    assert set(final_questions({"q": Noul("x")})) == {"q", SUFF + "q"}


def test_text_chunks_round_trip_and_respect_limit():
    text = "\n\n".join(f"Paragraph {i}. " + "word " * 120 for i in range(40)) + "\n" + "x" * 9000
    chunks = chunk_state(text, 200, EST)
    assert "".join(c.text for c in chunks) == text
    assert max(len(c.text) for c in chunks) <= 800
    assert [c.id for c in chunks] == list(range(len(chunks)))
    assert chunks[3].start == sum(len(c.text) for c in chunks[:3])
    assert chunks[0].label == "$#c0"


def test_list_and_dict_chunks_are_labelled():
    state = {"emails": [{"n": i, "body": "hello " * 30} for i in range(20)], "note": "short"}
    chunks = chunk_state(state, 100, EST)
    labels = [c.label for c in chunks]
    assert labels[0].startswith("$.emails[0:")
    assert labels[-1] == "$.note#c0"
    big = chunk_state(["tiny", "y" * 2000], 100, EST)
    assert any(c.label.startswith("$[1]#c") for c in big)


def test_estimator_observe_and_adaptive():
    fixed = TokenEstimator(4.0)
    fixed.observe(1000, 500)
    assert fixed.chars_per_token == 4.0 and fixed.observed_chars_per_token == 2.0
    adaptive = TokenEstimator(4.0, adaptive=True)
    adaptive.observe(1000, 500)
    assert adaptive.estimate("x" * 100) == 50


def _chunks(n=10, size=400):
    return chunk_state("".join(f"{i:03d}" + "a" * (size - 5) + "\n\n" for i in range(n)), 100, EST)


def test_round_robin_budget_order_and_fairness():
    chunks = _chunks()
    rel = {"q1": np.linspace(0, 1, len(chunks)), "q2": np.linspace(1, 0, len(chunks))}
    kept = select_round_robin(chunks, rel, budget_tokens=4 * (100 + 15))
    assert len(kept) == 4
    assert kept[0] == len(chunks) - 1 and kept[1] == 0  # each question gets its best first
    assert sum(chunks[i].est_tokens + 15 for i in kept) <= 4 * 115


def test_truncate_keeps_head_and_tail():
    chunks = _chunks(20)
    kept = select_truncate(chunks, 600)
    assert 0 in kept and len(chunks) - 1 in kept and len(chunks) // 2 not in kept


def test_render_marks_gaps_and_merges_neighbours():
    chunks = _chunks()
    text = render_state(chunks, [5, 1, 2])
    assert text.count("omitted") == 3  # before 1, between 2 and 5, after 5
    assert chunks[1].text + chunks[2].text in text
    assert text.index(chunks[1].text) < text.index(chunks[5].text)
    assert "omitted" not in render_state(chunks, list(range(len(chunks))))


def test_split_questions_respects_total_limit():
    qs = {f"q{i}": {"type": "noul", "instructions": "x" * 40_000} for i in range(12)}
    groups = split_questions(qs, 5_000, EST)
    assert len(groups) > 1 and sum(len(g) for g in groups) == 12


def test_vote_weights_by_relevance_and_sufficiency():
    n = 3
    scores = ChunkScores(
        rel={"q": np.array([0.9, 0.1, 0.0], dtype=np.float32), "b": np.array([0.9, 0.9, 0], dtype=np.float32)},
        suff={"q": np.array([0.9, 0.1, 0.0], dtype=np.float32), "b": np.array([1, 0.5, 0], dtype=np.float32)},
        ans={
            "q": [
                {"probabilities": {"A": 0.1, "B": 0.9}},
                {"probabilities": {"A": 0.9, "B": 0.1}},
                None,
            ],
            "b": [{"noul": 1.0}, {"noul": 0.0}, None],
        },
    )
    out = vote(scores, {"q": Choice("?", ["A", "B"]), "b": Noul("?")})
    assert out["q"]["choice"] == "B"
    assert out["b"]["noul"] == pytest.approx(0.9 / 1.35)
    empty = ChunkScores(rel={"q": np.zeros(n)}, suff={"q": np.zeros(n)}, ans={"q": [None] * n})
    assert vote(empty, {"q": Choice("?", ["A", "B"])})["q"] is None


def test_agrees():
    assert agrees({"type": "choice", "choice": "A"}, {"type": "choice", "choice": "A"}) is True
    assert agrees({"type": "noul", "noul": 0.9}, {"type": "noul", "noul": 0.2}) is False
    assert agrees({"type": "choice", "choice": "A"}, None) is None


def test_shortlist_minimum_and_fraction():
    rel = {"q": np.arange(400, dtype=np.float32)}
    ids = shortlist(rel)
    assert len(ids) == 100 and min(ids) == 300
    assert len(shortlist({"q": np.arange(30, dtype=np.float32)})) == 30


def test_cache_round_trip_and_key_stability(tmp_path):
    cache = Cache(tmp_path / "c.sqlite")
    k1 = request_key("decide", "m", "state", {"b": 1, "a": 2})
    k2 = request_key("decide", "m", "state", {"a": 2, "b": 1})
    assert k1 == k2 and k1 != request_key("decide", "m", "state2", {"a": 2, "b": 1})
    assert cache.get_response(k1) is None
    cache.put_response(k1, {"answers": {"x": 1}})
    cache.put_vector("v", np.array([1.0, 2.0]))
    cache.flush()
    again = Cache(tmp_path / "c.sqlite")
    assert again.get_response(k1) == {"answers": {"x": 1}}
    assert again.get_vector("v").tolist() == [1.0, 2.0]


def test_cache_writes_survive_without_flush(tmp_path):
    cache = Cache(tmp_path / "c.sqlite")
    cache.put_response("k", {"answers": {"x": 1}})  # no flush(), no close: a script that just exits
    assert Cache(tmp_path / "c.sqlite").get_response("k") == {"answers": {"x": 1}}


def test_dict_chunks_keep_their_field_names():
    chunks = chunk_state({"governing_law": "Delaware. " * 40, "notice": "30 days"}, 50, EST)
    assert chunks[0].text.startswith("governing_law: Delaware")
    assert chunks[-1].text == "notice: 30 days"
    assert not chunks[1].text.startswith("governing_law")  # only the first piece of a field is prefixed


def test_selection_is_never_empty():
    chunks = _chunks(5, size=400)
    rel = {"q": np.array([0.1, 0.9, 0.2, 0.3, 0.4], dtype=np.float32)}
    assert select_round_robin(chunks, rel, 10) == [1]
    assert select_truncate(chunks, 10) == [0]


def test_vote_tolerates_missing_probabilities_and_named_score_levels():
    one = np.ones(2, dtype=np.float32)
    scores = ChunkScores(
        rel={"c": one, "s": one}, suff={"c": one, "s": one},
        ans={
            "c": [{"choice": "A"}, {"probabilities": {"A": 0.2, "B": 0.8}}],
            "s": [{"probabilities": {"low": 0.3, "high": 0.7}}, None],
        },
    )
    out = vote(scores, {"c": Choice("?", ["A", "B"]), "s": Score("?", ["low", "high"])})
    assert out["c"]["choice"] == "B" and out["s"]["score"] == "high"
