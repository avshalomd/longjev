"""Question types and the questions longjev asks about each chunk."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Union


@dataclass(frozen=True)
class Noul:
    instructions: str
    criteria: Mapping[str, str] | None = None  # {"true": ..., "false": ...}

    def to_json(self) -> dict:
        body: dict[str, Any] = {"type": "noul", "instructions": self.instructions}
        if self.criteria:
            body["criteria"] = dict(self.criteria)
        return body


@dataclass(frozen=True)
class Choice:
    instructions: str
    options: Mapping[str, str | None] | tuple[str, ...] | list[str]

    def to_json(self) -> dict:
        options = self.options
        if not isinstance(options, Mapping):
            options = {o: None for o in options}
        return {"type": "choice", "instructions": self.instructions, "criteria": dict(options)}


@dataclass(frozen=True)
class Score:
    instructions: str
    levels: tuple[str, ...] | list[str]

    def to_json(self) -> dict:
        return {"type": "score", "instructions": self.instructions, "criteria": list(self.levels)}


Question = Union[Noul, Choice, Score, Mapping[str, Any]]

REL, ANS, SUFF = "rel__", "ans__", "suff__"


def to_json(question: Question) -> dict:
    """The API's JSON shape for a question object or an already-raw dict."""
    if isinstance(question, Mapping):
        if question.get("type") not in ("noul", "choice", "score"):
            raise ValueError(f"unknown question type: {question.get('type')!r}")
        return dict(question)
    return question.to_json()


def render(question: Question) -> str:
    """One line of text describing a task question, used inside other questions."""
    q = to_json(question)
    instructions = q["instructions"]
    text = instructions if isinstance(instructions, str) else str(instructions)
    criteria = q.get("criteria")
    if q["type"] == "choice" and criteria:
        options = "; ".join(f"{k}: {v}" if v else str(k) for k, v in criteria.items())
        return f"{text} Options: {options}"
    if q["type"] == "score" and criteria:
        return f"{text} Levels: {'; '.join(criteria)}"
    return text


def chunk_questions(task_questions: Mapping[str, Question]) -> dict[str, dict]:
    """Questions sent with every chunk: relevance, the task question, sufficiency."""
    out: dict[str, dict] = {}
    for key, question in task_questions.items():
        rendered = render(question)
        out[REL + key] = Noul(
            "This passage contains information that helps answer the following "
            f"question: {rendered}"
        ).to_json()
        out[ANS + key] = to_json(question)
        out[SUFF + key] = Noul(
            f"The following question can be answered from this passage alone: {rendered}"
        ).to_json()
    return out


def final_questions(task_questions: Mapping[str, Question]) -> dict[str, dict]:
    """Questions for the final judgment: the task questions plus sufficiency."""
    out: dict[str, dict] = {}
    for key, question in task_questions.items():
        out[key] = to_json(question)
        out[SUFF + key] = Noul(
            f"The text contains enough information to answer: {render(question)}"
        ).to_json()
    return out
