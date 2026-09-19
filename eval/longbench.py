"""LongBench v2 loader and the fixed dev/test split."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from longjev import Choice

DATA_URL = "https://huggingface.co/datasets/zai-org/LongBench-v2/resolve/main/data.json"
DATA_PATH = Path("data/longbench_v2.json")
DEV_SIZE = 100
LETTERS = ("A", "B", "C", "D")


def load(path: Path = DATA_PATH) -> list[dict]:
    items = json.loads(path.read_text(encoding="utf-8"))
    order = sorted(items, key=lambda x: hashlib.sha256(x["_id"].encode()).hexdigest())
    dev = {x["_id"] for x in order[:DEV_SIZE]}
    for item in items:
        item["split"] = "dev" if item["_id"] in dev else "test"
    return items


def question(item: dict) -> dict[str, Choice]:
    return {"q": Choice(item["question"].strip(), {L: item[f"choice_{L}"].strip() for L in LETTERS})}
