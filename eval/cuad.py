"""CUAD contracts as a long-input decision task: does this contract contain clause type X?

CUAD v1 (The Atticus Project, CC BY 4.0) labels 41 clause types in 510 contracts. A type is
present when lawyers highlighted at least one span for it. Only contracts too long for Jev's
window are used, and only clause types that are present in roughly a quarter to half of them.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from longjev import Noul

DATA_URL = "https://huggingface.co/datasets/theatticusproject/cuad/resolve/main/CUAD_v1/CUAD_v1.json"  # 40 MB
PATH = Path("data/CUAD_v1.json")
CATEGORIES = (
    "Termination For Convenience", "Minimum Commitment", "Exclusivity", "Ip Ownership Assignment",
    "Change Of Control", "Non-Compete", "Renewal Term", "Rofr/Rofo/Rofn", "Liquidated Damages",
    "No-Solicit Of Employees",
)


def key(category: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", category.lower()).strip("_")


def _category(question: str) -> str:
    return re.search(r'related to "(.+?)"', question).group(1)


def _details(question: str) -> str:
    return question.split("Details:", 1)[1].strip()


def load(n: int = 40, min_chars: int = 90_000) -> tuple[list[dict], dict[str, tuple[str, str]]]:
    """Returns (contracts, definitions). definitions maps key -> (name, CUAD's own description)."""
    data = json.loads(PATH.read_text())["data"]
    definitions: dict[str, tuple[str, str]] = {}
    contracts = []
    for doc in data:
        para = doc["paragraphs"][0]
        if len(para["context"]) <= min_chars:
            continue
        labels = {}
        for qa in para["qas"]:
            name = _category(qa["question"])
            if name in CATEGORIES:
                labels[key(name)] = not qa["is_impossible"]
                definitions.setdefault(key(name), (name, _details(qa["question"])))
        contracts.append({"id": doc["title"], "context": para["context"], "labels": labels})
    contracts.sort(key=lambda c: hashlib.sha256(c["id"].encode()).hexdigest())
    return contracts[:n], definitions


def chunk_question(name: str, details: str) -> Noul:
    return Noul(f'This passage of a contract contains a "{name}" clause. What counts: {details}')


def contract_question(name: str, details: str) -> Noul:
    return Noul(f'This contract contains a "{name}" clause. What counts: {details}')
