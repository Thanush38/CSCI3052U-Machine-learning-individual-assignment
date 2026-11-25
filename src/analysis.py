"""Accuracy tables, comparison with the paper, and the results chart."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .prompts import ALL_LEVELS

REF_DIR = Path(__file__).resolve().parent.parent / "results" / "reference"

# colour-blind safe
COLORS = {"Paper": "#2a78d6", "Ours: released questions": "#eb6834", "Ours: own questions": "#1baf7a"}


def _level_order(levels) -> list[str]:
    return [lv for lv in ALL_LEVELS if lv in set(levels)]


def bootstrap_ci(correct: np.ndarray, n_boot: int = 2000, seed: int = 42) -> tuple[float, float]:
    """95% bootstrap CI, in percent."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(correct), size=(n_boot, len(correct)))
    accs = correct[idx].mean(axis=1) * 100
    return float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))


def accuracy_table(pred: pd.DataFrame, by_language: bool = False) -> pd.DataFrame:
    keys = ["level"] + (["language"] if by_language else [])
    rows = []
    for k, g in pred.groupby(keys):
        k = k if isinstance(k, tuple) else (k,)
        lo, hi = bootstrap_ci(g["correct"].to_numpy())
        rows.append({**dict(zip(keys, k)), "n": len(g), "accuracy": 100 * g["correct"].mean(),
                     "ci_low": lo, "ci_high": hi,
                     "unparsed": int((g["pred"] == "NONE").sum())})
    out = pd.DataFrame(rows)
    out["level"] = pd.Categorical(out["level"], _level_order(out["level"]), ordered=True)
    return out.sort_values(keys).reset_index(drop=True)


def paper_reference(model: str = "qwen2.5-7b") -> tuple[pd.DataFrame, pd.DataFrame]:
    per_lang = pd.read_csv(REF_DIR / "paper_appendix_b_per_language.csv")
    overall = pd.read_csv(REF_DIR / "paper_table3_overall.csv")
    return per_lang[per_lang.model == model], overall[overall.model == model]


def compare_to_paper(pred: pd.DataFrame, model: str = "qwen2.5-7b") -> pd.DataFrame:
    """paper_matched = Appendix B numbers for our languages; paper_all18 = Table 3."""
    per_lang, overall = paper_reference(model)
    ours = accuracy_table(pred)
    counts = pred.drop_duplicates(["id"]).groupby("language").size()
    rows = []
    for _, r in ours.iterrows():
        ref = per_lang[per_lang.level == r.level].set_index("language")["accuracy"]
        langs = [l for l in counts.index if l in ref.index]
        matched = float(np.average(ref[langs], weights=counts[langs])) if langs else np.nan
        all18 = overall.loc[overall.level == r.level, "accuracy"]
        rows.append({
            "level": r.level, "n": r.n,
            "ours": round(r.accuracy, 2), "ours_95ci": f"[{r.ci_low:.1f}, {r.ci_high:.1f}]",
            "paper_matched": round(matched, 2),
            "delta_vs_matched": round(r.accuracy - matched, 2),
            "paper_all18": float(all18.iloc[0]) if len(all18) else np.nan,
            "within_ci": bool(r.ci_low <= matched <= r.ci_high),
        })
    return pd.DataFrame(rows)


def plot_comparison(series: dict[str, pd.Series], out_png: str | Path, title: str = "",
                    chance: float = 25.0):
    import matplotlib.pyplot as plt

    levels = _level_order(set().union(*[s.index for s in series.values()]))
    names = list(series)
    n = len(names)
    width = 0.8 / n
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=150)
    x = np.arange(len(levels))
    for i, name in enumerate(names):
        vals = [series[name].get(lv, np.nan) for lv in levels]
        bars = ax.bar(x + (i - (n - 1) / 2) * width, vals, width * 0.92,
                      color=COLORS.get(name, "#6250d6"), label=name, zorder=3)
        for b, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.0f}", ha="center",
                        va="bottom", fontsize=8, color="#3d3d3a")
    ax.axhline(chance, color="#888780", lw=1.2, ls="--", zorder=2, label=f"Random guess ({chance:.0f}%)")
    ax.set_xticks(x, levels)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlabel("Information given to the model")
    if title:
        ax.set_title(title, fontsize=11, loc="left")
    ax.grid(axis="y", color="#e5e4df", lw=0.8, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)
    return out_png
