"""Load the authors' released questions (Benchmark_multiple_choice/*.txt)."""
from __future__ import annotations

import json
import random
import re
from pathlib import Path

LETTERS = ["A", "B", "C", "D"]

# folder name -> name used in the paper
LANGUAGE_NAMES = {
    "Fwe": "Fwe", "Gyeli": "Gyeli", "Ik": "Ik", "Japhug": "Japhug",
    "Kagayanen": "Kagayanen", "Kalamang": "Kalamang", "Komnzo": "Komnzo",
    "Mauwake": "Mauwake", "Mehweb": "Mehweb", "Moloko": "Moloko",
    "Palula": "Palula", "Papuan_Malay": "Papuan Malay", "Pichi": "Pichi",
    "Rapa_Nui": "Rapa Nui", "Tuatschin": "Tuatschin", "Ulwa": "Ulwa",
    "Vamale": "Vamale", "Yauyos_Quecha": "Yauyos Quechua",
}

_Q_SPLIT = re.compile(r"^Question (\d+):\s*$", re.M)
_SENT = re.compile(r"^Sentence \(with missing item\):\s?(.*)$", re.M)
_GLOSS = re.compile(r"^Gloss \(with missing item\):\s?(.*)$", re.M)
_TRANS = re.compile(r"^The English translation of this sentence is:\s?(.*)$", re.M)
_KP = re.compile(r"^Here is a relevant knowledge point[^:]*:\s?(.*?)(?=^A: word:)", re.M | re.S)
_OPT = re.compile(r"^([A-D]): word: (.*?)\t\s*gloss: (.*)$", re.M)
_ANS = re.compile(r"^Correct Answer:\s*([A-D])", re.M)


def parse_question_block(block: str) -> dict | None:
    """Parse one question block (None if malformed)."""
    try:
        sentence = _SENT.search(block).group(1).strip()
        gloss = _GLOSS.search(block).group(1).strip()
        trans_m = _TRANS.search(block)
        translation = trans_m.group(1).strip() if trans_m else ""
        kp_m = _KP.search(block)
        kp = " ".join(kp_m.group(1).split()) if kp_m else ""
        opts = {l: {"word": w.strip(), "gloss": g.strip()} for l, w, g in _OPT.findall(block)}
        answer = _ANS.search(block).group(1)
    except AttributeError:
        return None
    if sorted(opts) != LETTERS:
        return None
    return {
        "sentence": sentence,
        "gloss": gloss,
        "translation": translation,
        "knowledge_point": kp,
        "options": opts,
        "answer": answer,
    }


def load_language(benchmark_dir: str | Path, folder: str) -> list[dict]:
    lang_dir = Path(benchmark_dir) / folder
    records = []
    for path in sorted(lang_dir.glob("*_questions.txt")):
        text = path.read_text(encoding="utf-8")
        parts = _Q_SPLIT.split(text)
        for num, body in zip(parts[1::2], parts[2::2]):
            rec = parse_question_block(body)
            if rec is None:
                continue
            rec.update(
                id=f"{folder}/{path.stem}#q{num}",
                language=LANGUAGE_NAMES.get(folder, folder),
                language_folder=folder,
                chapter_file=path.stem,
                source="linggym_released",
            )
            records.append(rec)
    return records


def load_benchmark(benchmark_dir: str | Path, languages: list[str] | None = None) -> list[dict]:
    benchmark_dir = Path(benchmark_dir)
    folders = languages or sorted(p.name for p in benchmark_dir.iterdir() if p.is_dir())
    out = []
    for f in folders:
        out.extend(load_language(benchmark_dir, f))
    return out


def sample_per_language(records: list[dict], n_per_language: int, seed: int) -> list[dict]:
    """Seeded sample of n questions per language."""
    rng = random.Random(seed)
    by_lang: dict[str, list[dict]] = {}
    for r in records:
        by_lang.setdefault(r["language"], []).append(r)
    sample = []
    for lang in sorted(by_lang):
        items = by_lang[lang]
        k = min(n_per_language, len(items))
        sample.extend(rng.sample(items, k))
    return sample


def save_jsonl(records: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
