"""Run a model on the questions: score A-D logits, sample like the paper, or guess randomly."""
from __future__ import annotations

import json
import platform
import random
import re
import time
from pathlib import Path

import pandas as pd

from .data import LETTERS
from .prompts import build_prompt

PRED_COLUMNS = ["id", "language", "source", "level", "mode", "model", "pred", "answer", "correct", "raw"]


def load_model(name: str, quantization: str = "4bit"):
    """quantization: '4bit', '8bit' or 'none'."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    kwargs = {"device_map": "auto"}
    if quantization in ("4bit", "8bit"):
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = (
            BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_compute_dtype=torch.float16)
            if quantization == "4bit" else BitsAndBytesConfig(load_in_8bit=True)
        )
    else:
        import transformers

        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        # transformers 5 renamed torch_dtype -> dtype
        key = "dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype"
        kwargs[key] = dtype
    model = AutoModelForCausalLM.from_pretrained(name, **kwargs)
    model.eval()
    return model, tok


def to_chat(tok, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    if getattr(tok, "chat_template", None):
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt + "\nAnswer:"


def letter_token_ids(tok) -> dict[str, list[int]]:
    ids = {}
    for L in LETTERS:
        cands = set()
        for variant in (L, " " + L):
            enc = tok.encode(variant, add_special_tokens=False)
            if len(enc) == 1:
                cands.add(enc[0])
        if not cands:
            raise ValueError(f"Tokenizer has no single token for {L!r}")
        ids[L] = sorted(cands)
    return ids


def score_batch(model, tok, prompts: list[str], letter_ids: dict[str, list[int]]) -> list[tuple[str, str]]:
    """Pick the letter with the highest next-token logit."""
    import torch

    texts = [to_chat(tok, p) for p in prompts]
    enc = tok(texts, return_tensors="pt", padding=True).to(model.device)
    # so left padding doesn't shift positions
    position_ids = (enc["attention_mask"].cumsum(-1) - 1).clamp(min=0)
    with torch.no_grad():
        logits = model(**enc, position_ids=position_ids).logits[:, -1, :].float()
    out = []
    for row in logits:
        scores = torch.stack([row[letter_ids[L]].max() for L in LETTERS])
        probs = torch.softmax(scores, dim=0).tolist()
        pred = LETTERS[int(scores.argmax())]
        out.append((pred, json.dumps({L: round(p, 4) for L, p in zip(LETTERS, probs)})))
    return out


_LETTER = re.compile(r"(?<![A-Za-z])([A-D])(?![A-Za-z])")


def parse_letter(text: str) -> str | None:
    m = _LETTER.search(text)
    return m.group(1) if m else None


def generate_batch(model, tok, prompts: list[str], max_new_tokens: int = 32,
                   temperature: float = 0.7, top_p: float = 0.9,
                   repetition_penalty: float = 1.1) -> list[tuple[str | None, str]]:
    """Sampling settings from Appendix A of the paper."""
    import torch

    texts = [to_chat(tok, p) for p in prompts]
    enc = tok(texts, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        gen = model.generate(**enc, do_sample=True, temperature=temperature, top_p=top_p,
                             repetition_penalty=repetition_penalty, max_new_tokens=max_new_tokens,
                             pad_token_id=tok.pad_token_id)
    replies = tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return [(parse_letter(r), r.strip()) for r in replies]


def _done_keys(out_csv: Path) -> set[tuple[str, str]]:
    if not out_csv.exists():
        return set()
    df = pd.read_csv(out_csv, usecols=["id", "level"])
    return set(zip(df["id"], df["level"]))


def run_eval(questions: list[dict], levels: list[str], out_csv: str | Path, mode: str = "score",
             model=None, tok=None, model_name: str = "", batch_size: int = 4, seed: int = 42,
             max_new_tokens: int = 32, progress: bool = True) -> pd.DataFrame:
    """Append predictions to out_csv after every batch; skips rows already there."""
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    done = _done_keys(out_csv)
    jobs = [(q, lv) for lv in levels for q in questions if (q["id"], lv) not in done]
    if done:
        print(f"Resuming: {len(done)} predictions already in {out_csv}, {len(jobs)} to go")

    rng = random.Random(seed)
    letter_ids = letter_token_ids(tok) if mode == "score" else None
    batches = [jobs[i:i + batch_size] for i in range(0, len(jobs), batch_size)]
    iterator = batches
    if progress:
        from tqdm.auto import tqdm
        iterator = tqdm(batches, desc=f"{mode}:{model_name or 'random'}")

    for batch in iterator:
        prompts = [build_prompt(q, lv) for q, lv in batch]
        if mode == "random":
            results = [(rng.choice(LETTERS), "") for _ in batch]
        elif mode == "score":
            results = score_batch(model, tok, prompts, letter_ids)
        elif mode == "generate":
            results = generate_batch(model, tok, prompts, max_new_tokens=max_new_tokens)
        else:
            raise ValueError(mode)
        rows = [{
            "id": q["id"], "language": q["language"], "source": q.get("source", ""), "level": lv,
            "mode": mode, "model": model_name or "random", "pred": pred if pred else "NONE",
            "answer": q["answer"], "correct": int(pred == q["answer"]), "raw": raw,
        } for (q, lv), (pred, raw) in zip(batch, results)]
        pd.DataFrame(rows, columns=PRED_COLUMNS).to_csv(
            out_csv, mode="a", header=not out_csv.exists(), index=False)
    return pd.read_csv(out_csv)


def environment_info() -> dict:
    info = {"python": platform.python_version(), "platform": platform.platform(),
            "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    for mod in ["torch", "transformers", "accelerate", "bitsandbytes", "sentence_transformers",
                "numpy", "pandas"]:
        try:
            info[mod] = __import__(mod).__version__
        except Exception:
            info[mod] = None
    try:
        import torch
        info["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except Exception:
        pass
    return info
