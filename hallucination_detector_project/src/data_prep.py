"""
Data preparation for claim-level hallucination detection on RAGTruth.

Loads RAGTruth's source_info.jsonl and response.jsonl, flattens every
source_info into a single plain-text "source document", segments each
LLM response into sentences (our "claim" granularity), and maps RAGTruth's
character-span hallucination annotations onto each sentence to produce a
binary gold label: 1 = hallucinated (unsupported/contradicted), 0 = supported.
"""
import json
import re
from pathlib import Path

import nltk

nltk.data.path.append("/root/nltk_data")
from nltk.tokenize import sent_tokenize

RAGTRUTH_DIR = Path("/home/claude/RAGTruth/dataset")


def flatten_source_info(rec):
    """Turn any source_info (str for Summary, dict for QA/Data2txt) into plain text."""
    task_type = rec["task_type"]
    info = rec["source_info"]

    if isinstance(info, str):
        return info.strip()

    if task_type == "QA":
        # info = {"question": ..., "passages": "passage 1:...\n\npassage 2:...\n\n"}
        q = info.get("question", "")
        passages = info.get("passages", "")
        return f"Question: {q}\n\n{passages}".strip()

    if task_type == "Data2txt":
        # Flatten the Yelp business JSON into readable "key: value" lines,
        # including any existing review text, so it reads like a real source doc.
        lines = []

        def walk(prefix, obj):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    walk(f"{prefix}{k}: " if not prefix else f"{prefix} > {k}: ", v)
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    walk(f"{prefix}[{i}] ", item)
            else:
                if obj not in (None, "", "null"):
                    lines.append(f"{prefix}{obj}")

        walk("", info)
        return "\n".join(lines)

    # Fallback: dump as JSON text
    return json.dumps(info)


def load_source_lookup():
    """source_id -> {"task_type":..., "source": ..., "text": flattened plain text}"""
    lookup = {}
    with open(RAGTRUTH_DIR / "source_info.jsonl") as f:
        for line in f:
            rec = json.loads(line)
            lookup[rec["source_id"]] = {
                "task_type": rec["task_type"],
                "source": rec["source"],
                "text": flatten_source_info(rec),
            }
    return lookup


def sentence_spans(text):
    """Sentence-tokenize `text` and return list of (start_char, end_char, sentence_str)
    using the *original* text's own offsets, so we can compare with RAGTruth's
    character-level hallucination spans."""
    spans = []
    cursor = 0
    for sent in sent_tokenize(text):
        # locate this sentence in the original text starting from cursor
        idx = text.find(sent, cursor)
        if idx == -1:  # tokenizer normalized whitespace; fall back to a loose search
            idx = text.find(sent.strip(), cursor)
        if idx == -1:
            continue
        start, end = idx, idx + len(sent)
        spans.append((start, end, text[start:end]))
        cursor = end
    return spans


def overlaps(a_start, a_end, b_start, b_end):
    return a_start < b_end and b_start < a_end


def build_claim_records(response_rec):
    """
    For a single RAGTruth response, return a list of claim dicts:
      {sentence, start, end, gold_label (0/1), hallucination_types}
    gold_label = 1 if this sentence overlaps ANY annotated hallucination span.
    """
    text = response_rec["response"]
    spans = sentence_spans(text)
    labels = response_rec.get("labels") or []

    claims = []
    for start, end, sent in spans:
        sent_clean = sent.strip()
        if len(sent_clean) < 3:
            continue
        hit_types = []
        for lab in labels:
            if overlaps(start, end, lab["start"], lab["end"]):
                hit_types.append(lab.get("label_type", "Unknown"))
        claims.append({
            "sentence": sent_clean,
            "start": start,
            "end": end,
            "gold_label": 1 if hit_types else 0,
            "hallucination_types": hit_types,
        })
    return claims


def iter_dataset(split):
    """Yield unified example dicts for the given split ('train' or 'test')."""
    source_lookup = load_source_lookup()
    with open(RAGTRUTH_DIR / "response.jsonl") as f:
        for line in f:
            rec = json.loads(line)
            if rec["split"] != split:
                continue
            if rec["quality"] != "good":
                continue  # skip truncated / incorrect_refusal generations
            src = source_lookup.get(rec["source_id"])
            if src is None:
                continue
            claims = build_claim_records(rec)
            if not claims:
                continue
            yield {
                "response_id": rec["id"],
                "source_id": rec["source_id"],
                "model": rec["model"],
                "task_type": src["task_type"],
                "source": src["source"],
                "source_text": src["text"],
                "response_text": rec["response"],
                "claims": claims,
            }


if __name__ == "__main__":
    n, n_claims, n_pos = 0, 0, 0
    for ex in iter_dataset("test"):
        n += 1
        n_claims += len(ex["claims"])
        n_pos += sum(c["gold_label"] for c in ex["claims"])
    print(f"test examples: {n}, claims: {n_claims}, hallucinated claims: {n_pos} "
          f"({100*n_pos/n_claims:.1f}%)")
