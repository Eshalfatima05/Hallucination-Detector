"""
Step 3 (retrieval) + Step 4 (entailment signal) as feature engineering.

Because this environment has no access to pretrained embedding/NLI model
weights (huggingface.co is not reachable from the sandbox), retrieval uses
classical TF-IDF cosine similarity, and "entailment" is approximated by a
handful of well-established lexical/statistical grounding signals (novel
n-gram ratio, numeric/entity consistency, TF-IDF support) that are then fed
into a supervised classifier trained on RAGTruth's own train split. This is
the same family of signal used by SelfCheckGPT-style and feature-based
hallucination baselines in the literature.
"""
import re
import math
from collections import Counter

import nltk
nltk.data.path.append("/root/nltk_data")
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

STOPWORDS = set(r"""
a an the this that these those is are was were be been being have has had do does did
will would shall should may might must can could of in on at to for from with by as
and or but if then than so not no nor it its it's he she they them his her their our
your you i we us me my mine yours ours theirs him himself herself itself themselves
about above after again against all am any because before below between both down
during each few further here how into more most once only other over own same some
such under until up very what when where which while who whom why
""".split())

NUM_RE = re.compile(r"\b\d[\d,]*\.?\d*%?\b")
# crude proper-noun / entity proxy: capitalized word (not sentence-initial word alone)
CAP_WORD_RE = re.compile(r"\b[A-Z][a-zA-Z]{2,}\b")


def split_source_chunks(source_text, task_type):
    """Split a flattened source document into retrieval chunks."""
    if task_type == "QA":
        # split on passage boundaries first, then sentences within each
        parts = re.split(r"\n\s*\n", source_text)
        chunks = []
        for p in parts:
            p = p.strip()
            if not p:
                continue
            chunks.extend(s.strip() for s in sent_tokenize(p) if len(s.strip()) > 2)
        return chunks or [source_text]
    if task_type == "Data2txt":
        # already "key: value" lines from flatten_source_info
        return [l.strip() for l in source_text.split("\n") if l.strip()]
    # Summary / generic prose
    return [s.strip() for s in sent_tokenize(source_text) if len(s.strip()) > 2]


def tokenize_words(text):
    return re.findall(r"[A-Za-z0-9']+", text.lower())


def content_words(text):
    return [w for w in tokenize_words(text) if w not in STOPWORDS and len(w) > 1]


class SourceIndex:
    """Per-source-document TF-IDF retrieval index over its chunks, built once
    and reused across every response generated for that same source."""

    def __init__(self, source_text, task_type):
        self.chunks = split_source_chunks(source_text, task_type)
        self.source_text_lower = source_text.lower()
        self.source_words = set(tokenize_words(source_text))
        self.source_numbers = set(NUM_RE.findall(source_text))
        self.source_caps = set(CAP_WORD_RE.findall(source_text))
        try:
            self.vectorizer = TfidfVectorizer(
                ngram_range=(1, 2), min_df=1, stop_words="english"
            )
            self.chunk_matrix = self.vectorizer.fit_transform(self.chunks)
            self.ok = True
        except ValueError:
            self.ok = False  # e.g. empty vocabulary

    def top_evidence(self, claim, k=3):
        if not self.ok or not self.chunks:
            return [], []
        q = self.vectorizer.transform([claim])
        sims = cosine_similarity(q, self.chunk_matrix)[0]
        order = sims.argsort()[::-1][:k]
        return [self.chunks[i] for i in order], [float(sims[i]) for i in order]


def claim_features(claim_sentence, source_index: SourceIndex, position_ratio: float,
                    task_type: str):
    """Build the feature vector (dict) for one claim sentence against its source."""
    top_chunks, top_sims = source_index.top_evidence(claim_sentence, k=3)
    max_sim = top_sims[0] if top_sims else 0.0
    mean_top3_sim = sum(top_sims) / len(top_sims) if top_sims else 0.0

    claim_content = content_words(claim_sentence)
    if claim_content:
        novel_ratio = sum(1 for w in claim_content if w not in source_index.source_words) / len(claim_content)
    else:
        novel_ratio = 0.0

    claim_numbers = set(NUM_RE.findall(claim_sentence))
    if claim_numbers:
        num_mismatch_ratio = sum(1 for n in claim_numbers if n not in source_index.source_numbers) / len(claim_numbers)
    else:
        num_mismatch_ratio = 0.0

    claim_caps = set(CAP_WORD_RE.findall(claim_sentence))
    if claim_caps:
        cap_mismatch_ratio = sum(1 for c in claim_caps if c not in source_index.source_caps) / len(claim_caps)
    else:
        cap_mismatch_ratio = 0.0

    return {
        "max_sim": max_sim,
        "mean_top3_sim": mean_top3_sim,
        "novel_word_ratio": novel_ratio,
        "num_mismatch_ratio": num_mismatch_ratio,
        "has_numbers": float(bool(claim_numbers)),
        "cap_mismatch_ratio": cap_mismatch_ratio,
        "has_caps": float(bool(claim_caps)),
        "claim_len_words": float(len(claim_content)),
        "position_ratio": position_ratio,
        "is_qa": float(task_type == "QA"),
        "is_summary": float(task_type == "Summary"),
        "is_data2txt": float(task_type == "Data2txt"),
    }, top_chunks, top_sims


FEATURE_NAMES = [
    "max_sim", "mean_top3_sim", "novel_word_ratio", "num_mismatch_ratio",
    "has_numbers", "cap_mismatch_ratio", "has_caps", "claim_len_words",
    "position_ratio", "is_qa", "is_summary", "is_data2txt",
]
