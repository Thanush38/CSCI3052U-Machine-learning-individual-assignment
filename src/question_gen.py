"""Our own question generator, based on Section 4.2 of the paper.

Distractors: longest common substring of the gloss, nearest SBERT gloss, random word from the same chapter."""
from __future__ import annotations

import random
import re
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd

from .data import LANGUAGE_NAMES

SBERT_MODEL = "paraphrase-MiniLM-L6-v2"
PUNCT = ".,;:!?\"'‘’“”«»()[]{}`"

_BOLD_MULTI = re.compile(r"\\textbf\{([^{}]*\s[^{}]*)\}")


def _split_bold_spans(s: str) -> str:
    # \textbf{a b} -> \textbf{a} \textbf{b}
    return _BOLD_MULTI.sub(lambda m: " ".join(f"\\textbf{{{w}}}" for w in m.group(1).split()), s)


def clean_latex(s: str) -> str:
    s = re.sub(r"\\(ref|REF|sectref|tabref|figref|Sectref|cref)\{[^{}]*\}", "[ref]", s)
    s = re.sub(r"\\textsc\{([^{}]*)\}", lambda m: m.group(1).upper(), s)
    for _ in range(3):  # \cmd{text} -> text
        s = re.sub(r"\\[A-Za-z]+\*?\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\(redp)\b", "~", s)
    s = re.sub(r"\\[A-Za-z]+\*?", "", s)
    s = s.replace("{", "").replace("}", "").replace("\\", "")
    return " ".join(s.split())


def clean_word(tok: str) -> str:
    return clean_latex(tok).strip(PUNCT).lower()


def clean_gloss(tok: str) -> str:
    # keep inner dots (3SG.OBJ), drop trailing punctuation ("talk." -> "talk")
    g = clean_latex(tok).strip("\"'‘’“”«»()[]`")
    return g.rstrip(".,;:!?") or g


def _fields(content: str) -> dict:
    out = {}
    for part in content.split("|"):
        part = part.strip()
        m = re.match(r"^\\(gsrc|gll|gls|glt)\b\s*(.*)$", part, re.S)
        if m and m.group(1) not in out:
            out[m.group(1)] = m.group(2).strip()
    return out


def parse_igt_examples(csv_dir: str | Path, folder: str) -> list[dict]:
    """IGT examples where words and glosses line up, with their KP and chapter."""
    examples = []
    for path in sorted(Path(csv_dir, folder).glob("*.csv")):
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            kp = row.get("Knowledge Point")
            kp = "" if pd.isna(kp) else str(kp)
            for i in range(1, 11):
                content = row.get(f"Content {i}")
                if not isinstance(content, str) or content.strip().lower() == "not found":
                    continue
                f = _fields(content)
                if "gll" not in f or "gls" not in f:
                    continue
                words_raw = _split_bold_spans(f["gll"]).split()
                glosses_raw = f["gls"].split()
                if len(words_raw) != len(glosses_raw) or not words_raw:
                    continue
                bold = ["\\textbf" in w for w in words_raw]
                # Fwe bolds the \gsrc line instead of \gll
                if not any(bold) and "gsrc" in f:
                    src = _split_bold_spans(f["gsrc"]).split()
                    if len(src) == len(words_raw):
                        bold = ["\\textbf" in w for w in src]
                examples.append({
                    "chapter": path.stem,
                    "knowledge_point": clean_latex(kp),
                    "words": [clean_latex(w) for w in words_raw],
                    "glosses": [clean_gloss(g) for g in glosses_raw],
                    "bold": bold,
                    "translation": clean_latex(f.get("glt", "")),
                })
    return examples


def lcs_len(a: str, b: str) -> int:
    return SequenceMatcher(None, a, b, autojunk=False).find_longest_match(0, len(a), 0, len(b)).size


class GlossEmbedder:
    """SBERT embeddings of glosses (TF-IDF if the model can't be downloaded)."""

    def __init__(self, glosses: list[str], use_sbert: bool = True):
        self.glosses = glosses
        self.index = {g: i for i, g in enumerate(glosses)}
        self.backend = "tfidf"
        emb = None
        if use_sbert:
            try:
                from sentence_transformers import SentenceTransformer

                model = SentenceTransformer(SBERT_MODEL)
                emb = model.encode(glosses, batch_size=256, show_progress_bar=False)
                self.backend = "sbert"
            except Exception as e:
                print(f"[question_gen] SBERT unavailable ({type(e).__name__}); using TF-IDF fallback")
        if emb is None:
            from sklearn.feature_extraction.text import TfidfVectorizer

            emb = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4)).fit_transform(glosses).toarray()
        emb = np.asarray(emb, dtype=np.float32)
        emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
        self.emb = emb

    def ranked(self, gloss: str) -> list[str]:
        sims = self.emb @ self.emb[self.index[gloss]]
        return [self.glosses[j] for j in np.argsort(-sims, kind="stable")]


def _mask(text: str, terms: list[str]) -> str:
    for t in sorted({t for t in terms if len(t) >= 2}, key=len, reverse=True):
        text = re.sub(rf"(?<!\w){re.escape(t)}(?!\w)", "___", text, flags=re.I)
    return text


def generate_questions(csv_dir: str | Path, folder: str, seed: int = 42,
                       max_questions: int | None = None, use_sbert: bool = True) -> list[dict]:
    rng = random.Random(seed)
    language = LANGUAGE_NAMES.get(folder, folder)
    examples = parse_igt_examples(csv_dir, folder)

    # every (word, gloss) pair in the grammar -> chapters it appears in
    pool: dict[tuple[str, str], set[str]] = {}
    for ex in examples:
        for w, g in zip(ex["words"], ex["glosses"]):
            cw = clean_word(w)
            if cw and g and any(ch.isalnum() for ch in cw) and "..." not in g:
                pool.setdefault((cw, g), set()).add(ex["chapter"])
    pairs = sorted(pool)
    gloss_to_words: dict[str, list[str]] = {}
    for w, g in pairs:
        gloss_to_words.setdefault(g, []).append(w)
    glosses = sorted(gloss_to_words)

    # one question per bold word
    targets = [(ei, j) for ei, ex in enumerate(examples) for j, b in enumerate(ex["bold"]) if b]
    if max_questions is not None and len(targets) > max_questions:
        targets = sorted(rng.sample(targets, max_questions))

    embedder = GlossEmbedder(glosses, use_sbert=use_sbert) if targets else None
    questions = []
    for qn, (ei, j) in enumerate(targets):
        ex = examples[ei]
        word, gloss = clean_word(ex["words"][j]), ex["glosses"][j]
        if (word, gloss) not in pool:
            continue
        used_w, used_g = {word}, {gloss}

        def ok(w, g):
            return w not in used_w and g not in used_g

        # 1) longest common substring with the correct gloss
        scored = sorted(((lcs_len(gloss, g), rng.random(), g) for g in glosses if g not in used_g),
                        reverse=True)
        lcs = next(((rng.choice(ws), g) for _, _, g in scored
                    if (ws := [w for w in gloss_to_words[g] if w not in used_w])), None)
        if lcs is None:
            continue
        lcs_w, lcs_g = lcs
        used_w.add(lcs_w); used_g.add(lcs_g)

        # 2) nearest gloss by SBERT
        sem = next(((w, g) for g in embedder.ranked(gloss) for w in gloss_to_words[g] if ok(w, g)), None)
        if sem is None:
            continue
        used_w.add(sem[0]); used_g.add(sem[1])

        # 3) random pair from the same chapter
        local = [p for p in pairs if ex["chapter"] in pool[p] and ok(*p)]
        if not local:
            local = [p for p in pairs if ok(*p)]
        chap = rng.choice(local)

        opts = [("gold", word, gloss), ("lcs", lcs_w, lcs_g), ("semantic", *sem), ("chapter", *chap)]
        rng.shuffle(opts)
        letters = ["A", "B", "C", "D"]
        sentence = ex["words"][:]
        sentence[j] = "___"
        gloss_line = ex["glosses"][:]
        gloss_line[j] = "___"
        questions.append({
            "id": f"{folder}/own/{ex['chapter']}#q{qn}",
            "language": language,
            "language_folder": folder,
            "chapter_file": ex["chapter"],
            "source": "own_generator",
            "sentence": " ".join(sentence),
            "gloss": " ".join(gloss_line),
            "translation": ex["translation"],
            "knowledge_point": _mask(ex["knowledge_point"], [word, gloss]),
            "options": {l: {"word": o[1], "gloss": o[2]} for l, o in zip(letters, opts)},
            "distractor_types": {l: o[0] for l, o in zip(letters, opts)},
            "answer": letters[[o[0] for o in opts].index("gold")],
            "embedding_backend": embedder.backend,
        })
    return questions
