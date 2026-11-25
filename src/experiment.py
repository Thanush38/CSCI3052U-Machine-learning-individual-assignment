"""Run one experiment and save everything to results/<run>/."""
from __future__ import annotations

import json
from pathlib import Path

from .analysis import accuracy_table
from .data import load_jsonl
from .evaluate import environment_info, run_eval
from .prompts import ALL_LEVELS, MAIN_LEVELS
from .seed import set_seed

ROOT = Path(__file__).resolve().parent.parent


def resolve_levels(levels: str | list[str]) -> list[str]:
    if isinstance(levels, list):
        return levels
    return {"main": MAIN_LEVELS, "all": ALL_LEVELS}.get(levels) or levels.split(",")


def run_experiment(questions: str | Path, run: str, mode: str = "score", model=None, tok=None,
                   model_name: str = "random", levels: str | list[str] = "main",
                   n_per_language: int | None = None, limit: int | None = None,
                   batch_size: int = 4, seed: int = 42, max_new_tokens: int = 32,
                   extra_config: dict | None = None):
    set_seed(seed)
    levels = resolve_levels(levels)
    qs = load_jsonl(ROOT / questions if not Path(questions).is_absolute() else questions)
    if n_per_language is not None:
        seen: dict[str, int] = {}
        kept = []
        for q in qs:
            if seen.get(q["language"], 0) < n_per_language:
                kept.append(q)
                seen[q["language"]] = seen.get(q["language"], 0) + 1
        qs = kept
    qs = qs[:limit]

    out_dir = ROOT / "results" / run
    out_dir.mkdir(parents=True, exist_ok=True)
    config = {"questions": str(questions), "run": run, "mode": mode, "model": model_name,
              "levels": levels, "n_questions": len(qs), "batch_size": batch_size, "seed": seed,
              "max_new_tokens": max_new_tokens, **(extra_config or {})}
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    (out_dir / "environment.json").write_text(json.dumps(environment_info(), indent=2))

    pred = run_eval(qs, levels, out_dir / "predictions.csv", mode=mode, model=model, tok=tok,
                    model_name=model_name, batch_size=batch_size, seed=seed,
                    max_new_tokens=max_new_tokens)
    acc = accuracy_table(pred)
    acc.to_csv(out_dir / "accuracy.csv", index=False)
    accuracy_table(pred, by_language=True).to_csv(out_dir / "accuracy_by_language.csv", index=False)
    return pred, acc
