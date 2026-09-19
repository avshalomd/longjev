# longjev — version 1 design

Date: 2026-09-18. Status: awaiting review by Avshalom.

## 1. Purpose

Jev (TypeSafe AI's decision model) accepts at most 32K tokens of input.
`longjev` is a Python library with the same call shape as Jev's
`system_one(state, questions)` that accepts a state of any length. It uses Jev
itself to keep the parts of the state that matter for the questions, then asks
Jev the original questions over the reduced state.

Version 1 ships two things: the library, and a published benchmark that shows
whether selecting with Jev beats selecting with a strong embedding model.

## 2. Background

- Limits (TypeSafe models page): state plus the longest single question must
  fit in 32K tokens; state plus all questions in 64K. Text only.
- TypeSafe's own weaknesses page says accuracy falls as the state grows
  ("context rot") and advises filtering first, for example with a relevance
  Noul. So a smaller, cleaner state should help even when the input fits.
- TypeSafe publishes no long-input wrapper and no long-context benchmark. Its
  cookbooks show small recipes only (see `ideas.md`, "Prior art checked").
- `tamaratran/fast-jev-compaction` solves this for agent sessions only, never
  lets Jev read the content it drops, and fails when its skeleton exceeds 25K
  tokens.

## 3. Goals and non-goals

Goals:

1. Drop-in call: `LongJev().system_one(state, questions)` returns Jev's answer
   shape plus a diagnostics block.
2. Any state length; inputs that already fit are passed straight through.
3. A reproducible LongBench v2 benchmark comparing four selectors at equal
   budget, with cost and latency.
4. Simple enough to build quickly. Total benchmark spend about $10.

Non-goals for version 1: everything in section 12.

## 4. Public API

```python
from longjev import LongJev, Noul, Choice, Score

lj = LongJev()                      # reads OPENROUTER_API_KEY
result = lj.system_one(state=long_text, questions={
    "breach": Noul("Does the contract allow termination without notice?"),
    "law": Choice("Which law governs?", {"NY": None, "DE": None, "other": None}),
})
result.answers["law"].choice        # same fields as Jev's answers
result.long_input                   # diagnostics, see 5.9
```

- `state`: `str`, `list` or `dict`, as in Jev.
- `questions`: map of key to `Noul`, `Choice` or `Score`, or to a raw dict in
  the API's JSON shape (`type`, `instructions`, optional or required
  `criteria`).
- Constructor options: `model` (default `typesafe/jev-latest`), `selector`
  (default `"jev"`), `budget_tokens` (default set from the benchmark, 16K until
  then), `max_budget_tokens` (28,000), `chunk_tokens` (800), `concurrency`
  (16), `cache` (off by default), `transport` (default OpenRouter).
- `asystem_one` is the async version; `system_one` wraps it.

## 5. Components

Each is one module under `src/longjev/` with one job and a narrow interface.

### 5.1 Transport (`transport.py`)

`Transport.decide(state, questions, model) -> dict` returns Jev's response
JSON. `OpenRouterTransport` posts to
`https://openrouter.ai/api/alpha/decisions` with a Bearer key, using `httpx`.
The endpoint is alpha, so nothing else in the library knows about it. A direct
TypeSafe transport can be added later without other changes.

Retries: exponential backoff with jitter on 429, 529 and network errors, up to
6 attempts. 401 and other 4xx raise immediately.

### 5.2 Cache (`cache.py`)

SQLite file keyed by SHA-256 of endpoint, model and canonical request JSON.
Stores the response and usage. Off by default in the library; always on in the
benchmark, which also pins `typesafe/jev-1.13` so cached results stay valid.

### 5.3 Token estimate (`tokens.py`)

Tokens are estimated as characters × ratio. The ratio starts conservative
(1 token per 3 characters) and is updated from `usage.input_tokens` in real
responses. The assembler keeps a 10% margin. If Jev still rejects a request as
too long, the assembler drops the lowest-scored chunk and retries, at most 3
times.

### 5.4 Chunker (`chunker.py`)

- `str`: split at blank lines, pack paragraphs greedily up to `chunk_tokens`.
  An oversized paragraph is split at sentence ends, then hard-split.
- `list`: items are serialised and packed greedily; an oversized item is
  chunked as text.
- `dict`: each top-level value is chunked by its own type.

Every chunk carries `id` (sequence number), `label` (JSON path plus chunk
index, e.g. `$.emails[12:15]` or `$#c7`), character offsets, text and estimated
tokens. Joining all chunks of a string state reproduces it exactly. The `label`
field exists so version 2 can attach real structural labels.

### 5.5 Question builder (`questions.py`)

Holds `Noul`, `Choice`, `Score` and renders any task question to one line of
text (instructions plus options or levels). For each chunk request it builds,
per task question `k`:

- `rel_k` (Noul): "This passage contains information that helps answer the
  following question: {rendered question}".
- `ans_k`: the task question itself, unchanged.
- `suff_k` (Noul): "The following question can be answered from this passage
  alone: {rendered question}".

If the questions for one chunk would break the 64K total, they are split over
several requests with the same state.

### 5.6 Selectors (`selectors.py`)

Each returns a relevance score per chunk per task question.

| Name | How it scores |
|---|---|
| `truncate` | No scoring. Keeps the first and last half of the budget, which is LongBench's own convention. The floor. |
| `embed` | Cosine similarity with `qwen/qwen3-embedding-8b` through OpenRouter (`/api/v1/embeddings`). The query is the rendered question with Qwen's retrieval instruction prefix. |
| `jev` | One Jev request per chunk; score is `rel_k`. Every chunk is read, so nothing is lost before scoring. |
| `embed_then_jev` | `embed` picks a shortlist (top 25% of chunks, at least 40); Jev scores only the shortlist. The cheap mode. In the benchmark it is computed from `jev`'s cached scores at no extra cost. |

### 5.7 Assembler (`assembler.py`)

Fills the budget by round-robin over task questions: each question in turn
takes its best unselected chunk, until the budget is full. Selected chunks are
put back in original order, adjacent chunks are merged, and each gap becomes a
`[… about N tokens omitted …]` line. The result is a string. It enforces the
32K and 64K limits with the margin from 5.3.

All selectors fill the same budget, so the comparison is fair.

### 5.8 Reducer (`reducer.py`)

Two results from one scoring pass:

- **Final judgment (the returned answer).** One Jev request with the assembled
  state, the task questions, and per question a sufficiency Noul: "The text
  contains enough information to answer: {rendered question}". If any
  sufficiency is below 0.5 and the budget is below `max_budget_tokens`, widen
  once to the maximum and ask again.
- **Vote (diagnostic).** Available when Jev scored the chunks. Each chunk's
  `ans_k` distribution is weighted by `rel_k × suff_k` and summed. Choice and
  Score take the top of the combined distribution; Noul takes the weighted
  mean. If total weight is near zero the vote is `None`.

`agreement[k]` says whether the two match.

### 5.9 Result

`answers` (Jev's shape), `model`, `usage` (summed over all requests), and
`long_input`: `passthrough`, `selector`, `n_chunks`, `kept` (id, label, score),
`state_tokens`, `widened`, `sufficiency`, `vote`, `agreement`,
`failed_chunks`, `n_requests`, `est_cost_usd`.

### 5.10 Score log

Every chunk score is appended to a JSONL file when a log path is set: document
hash, chunk id, label, offsets, question key, `rel`, `suff`, answer
distribution. The benchmark always sets it. Version 2's setup step would read
this.

## 6. Data flow

1. Estimate size. If state and questions fit in `max_budget_tokens`, call Jev
   directly and return with `passthrough=True`.
2. Chunk the state.
3. Score chunks with the chosen selector, concurrently.
4. Assemble to `budget_tokens`.
5. Final judgment; widen once if sufficiency is low.
6. Compute the vote and agreement; return.

## 7. Errors

- Missing key: clear error naming `OPENROUTER_API_KEY`.
- A chunk request that fails after all retries: the chunk gets score 0, is
  listed in `long_input.failed_chunks`, and the call continues. If more than
  10% of chunks fail, raise.
- Final judgment failure: raise. There is no silent fallback answer.

## 8. Benchmark (`eval/`)

**Dataset.** LongBench v2 from Hugging Face (`zai-org/LongBench-v2`): 503
four-option questions, contexts from 8K to 2M words, six categories, Apache-2.0.
Each item becomes one `Choice` with options A–D. Reference points: guessing
25%, human experts 53.7%, LLM scores quoted from the public leaderboard.

**Split.** Items sorted by SHA-256 of `_id`; the first 100 are the dev split,
the other 403 the test split.

**Procedure.**

1. One Jev scoring pass over every chunk of every item (cached).
2. On dev, run each selector at budgets 8K, 16K and 28K. Freeze the best
   budget per selector.
3. On test, run `truncate`, `embed`, `jev`, `embed_then_jev` at their frozen
   budgets, plus the vote on its own.
4. Control group: for items that fit in 28K, compare direct Jev against each
   selector at 8K. This measures what compression costs, and tests the
   context-rot claim (compression may help).

**Primary comparison, fixed in advance:** `jev` against `embed` on the test
split, McNemar's exact test on paired outcomes.

**Reported.** Accuracy with 95% Wilson intervals, overall and by length,
difficulty and category; accuracy when final judgment and vote agree against
when they disagree; dollars, requests, tokens and wall-clock time per item.
Results are reported whichever way they fall.

**Floor-effect rule.** If every selector lands within its interval of 30%,
LongBench v2 cannot separate them. In that case BABILong moves from deferred to
next.

**Spend control.** `--limit N` for a first 5-item run to check the cost
estimate; `--max-usd` aborts when spend computed from usage exceeds the cap.
Estimate: scoring pass about $5 (about 120M tokens at $0.042/M), embeddings
about $1, final calls about $2.50.

## 9. Tests

`pytest` with a `FakeTransport` that returns scripted probabilities. Covered:
chunker round-trip and boundaries; token estimate updates; assembler budget,
order, gap lines and limits; question splitting at 64K; vote arithmetic;
widening; cache hits and key stability; retry on 429/529 and immediate raise on
401; pass-through; failed-chunk handling. One live smoke test, skipped without
`OPENROUTER_API_KEY`, costing under $0.001.

## 10. Layout and tooling

```
longjev/
  pyproject.toml          uv, Python ≥ 3.11
  README.md  ideas.md  .env.example  .gitignore
  src/longjev/            api, questions, chunker, tokens, transport,
                          cache, embed, selectors, assembler, reducer
  eval/                   longbench.py, run.py, report.py
  tests/
  docs/superpowers/specs/
```

Runtime dependencies: `httpx`, `numpy`. Benchmark extras: `datasets`,
`python-dotenv`. Dev: `pytest`, `pytest-asyncio`.

## 11. Data handling

The state is sent to OpenRouter and on to TypeSafe; with the `embed` selectors
it is also sent to the embedding provider. The README says so plainly. The key
lives in a gitignored `.env` and is never logged. The cache and score log hold
full input text and are gitignored.

## 12. Out of version 1

Recorded in `ideas.md`: the version 2 task-specific compiled reader; the
structure ablation; tree-shaped chunks, structure detection and
dangling-reference expansion; local embeddings; reranker baseline; BABILong;
LongMemEval; global arbitration pass; JSON field pruning; top-down navigation;
typed index; TypeScript port. Also out: adaptive budget by relevance threshold,
and context pinned into every chunk request for `dict` states.

## 13. Risks

- LongBench v2 is reasoning-heavy; Jev may sit near the floor. Handled by the
  floor-effect rule.
- n=403 gives about ±5 points, so small differences will not be significant.
- The OpenRouter endpoint is alpha and may change. Isolated in the transport.
- TypeSafe calls multi-hop questions a weakness. Selection cannot fix what the
  final judgment cannot do.

## 14. For Avshalom to confirm at review

1. Name `longjev`.
2. Licence: MIT recommended (same as fast-jev-compaction; LongBench v2 is only
   downloaded, not redistributed).
3. Default selector `jev` (most accurate in principle, most expensive) until
   the benchmark says otherwise. The alternative default is `embed_then_jev`.
4. The two small additions for version 2 readiness are in (chunk `label`, score
   log). The three larger structural ones are out, to keep version 1 quick.
