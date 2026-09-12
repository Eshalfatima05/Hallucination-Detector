import json
import time
from pathlib import Path

from src.data_prep import iter_dataset
from src.features import SourceIndex, claim_features, FEATURE_NAMES

OUT_DIR = Path("/home/claude/project/results")
OUT_DIR.mkdir(exist_ok=True, parents=True)


def build_split(split):
    rows = []
    source_index_cache = {}
    t0 = time.time()
    n_examples = 0
    for ex in iter_dataset(split):
        n_examples += 1
        src_id = ex["source_id"]
        if src_id not in source_index_cache:
            source_index_cache[src_id] = SourceIndex(ex["source_text"], ex["task_type"])
        idx = source_index_cache[src_id]

        n_claims = len(ex["claims"])
        for i, c in enumerate(ex["claims"]):
            position_ratio = i / max(n_claims - 1, 1)
            feats, top_chunks, top_sims = claim_features(
                c["sentence"], idx, position_ratio, ex["task_type"]
            )
            rows.append({
                "response_id": ex["response_id"],
                "source_id": src_id,
                "task_type": ex["task_type"],
                "model": ex["model"],
                "sentence": c["sentence"],
                "gold_label": c["gold_label"],
                "hallucination_types": c["hallucination_types"],
                "top_evidence": top_chunks,
                "top_sims": top_sims,
                "features": feats,
            })
        if n_examples % 2000 == 0:
            print(f"[{split}] {n_examples} examples, {len(rows)} claims, "
                  f"{time.time()-t0:.1f}s elapsed")

    print(f"[{split}] DONE: {n_examples} examples -> {len(rows)} claims "
          f"in {time.time()-t0:.1f}s")
    out_path = OUT_DIR / f"claims_{split}.jsonl"
    with open(out_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {out_path}")
    return rows


if __name__ == "__main__":
    build_split("test")
    build_split("train")
