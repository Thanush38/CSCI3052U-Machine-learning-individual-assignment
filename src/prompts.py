"""Prompt template from Figure 4 of the paper. Levels: S, S+G, S+G+KP, S+G+KP+T."""
from __future__ import annotations

MAIN_LEVELS = ["S", "S+G", "S+G+KP", "S+G+KP+T"]
ALL_LEVELS = ["S", "S+G", "S+T", "S+KP", "S+G+KP", "S+G+T", "S+KP+T", "S+G+KP+T"]


def parse_level(level: str) -> set[str]:
    parts = set(level.split("+"))
    if "S" not in parts or not parts <= {"S", "G", "KP", "T"}:
        raise ValueError(f"Bad level {level!r}; expected e.g. 'S+G+KP'")
    return parts


def _given_phrase(parts: set[str]) -> str:
    extras = []
    if "G" in parts:
        extras.append("gloss")
    if "T" in parts:
        extras.append("translation")
    if not extras:
        return "a sentence along with its morpheme breakdown"
    if len(extras) == 1:
        return f"a sentence along with its morpheme breakdown and {extras[0]}"
    return "a sentence along with its morpheme breakdown, gloss and translation"


def build_prompt(q: dict, level: str) -> str:
    parts = parse_level(level)
    missing = "a word and its gloss are missing" if "G" in parts else "one word is missing"
    lines = [
        f"You are a linguist specializing in {q['language']}. "
        f"You are given {_given_phrase(parts)}. "
        "Words are separated by spaces, and morphemes are separated by hyphens. "
        f"However, {missing} and represented by an underscore. "
        "Based on your understanding, please choose the most appropriate option.",
        f"Sentence (with missing item): {q['sentence']}",
    ]
    if "G" in parts:
        lines.append(f"Gloss (with missing item): {q['gloss']}")
    if "T" in parts:
        lines.append(f"The English translation of this sentence is: {q['translation']}")
    if "KP" in parts:
        lines.append(
            "Here is a relevant knowledge point for this example, with the related "
            f"morphemes and glosses masked: {q['knowledge_point']}"
        )
    lines.append("Options:")  # always word + gloss, as in the paper's Fig. 2 and 4
    for letter in ["A", "B", "C", "D"]:
        o = q["options"][letter]
        lines.append(f"{letter}: word: {o['word']}    gloss: {o['gloss']}")
    lines.append("Please only return the letter (A–D). Do not output anything else.")
    return "\n".join(lines)
