import json
from pathlib import Path
from collections import Counter

RESULTS_DIR = Path("/home/claude/project/results")


def load_predictions():
    rows = []
    with open(RESULTS_DIR / "test_predictions.jsonl") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def main():
    rows = load_predictions()

    fn = [r for r in rows if r["gold_label"] == 1 and r["pred_label"] == 0]  # missed hallucinations
    fp = [r for r in rows if r["gold_label"] == 0 and r["pred_label"] == 1]  # false alarms
    tp = [r for r in rows if r["gold_label"] == 1 and r["pred_label"] == 1]

    print(f"False negatives (missed hallucinations): {len(fn)}")
    print(f"False positives (false alarms): {len(fp)}")
    print(f"True positives (caught hallucinations): {len(tp)}")

    # breakdown of missed hallucinations by RAGTruth's own label_type taxonomy
    fn_types = Counter()
    for r in fn:
        for t in r["hallucination_types"]:
            fn_types[t] += 1
    tp_types = Counter()
    for r in tp:
        for t in r["hallucination_types"]:
            tp_types[t] += 1

    print("\nMissed (FN) hallucination types:", dict(fn_types))
    print("Caught (TP) hallucination types:", dict(tp_types))

    all_types = set(fn_types) | set(tp_types)
    print("\nRecall by hallucination type:")
    type_recall = {}
    for t in all_types:
        total = fn_types[t] + tp_types[t]
        rec = tp_types[t] / total if total else 0
        type_recall[t] = {"n": total, "recall": rec}
        print(f"  {t}: n={total}, recall={rec:.2f}")

    # length effect
    def avg_len(rs):
        lens = [r["features"]["claim_len_words"] for r in rs]
        return sum(lens) / len(lens) if lens else 0

    print(f"\nAvg claim length (content words): FN={avg_len(fn):.1f}, TP={avg_len(tp):.1f}, "
          f"FP={avg_len(fp):.1f}")

    # novel_word_ratio distribution for FN vs TP -- do missed hallucinations "look" grounded?
    def avg_feat(rs, name):
        vals = [r["features"][name] for r in rs]
        return sum(vals) / len(vals) if vals else 0

    for feat in ["novel_word_ratio", "max_sim", "num_mismatch_ratio"]:
        print(f"avg {feat}: FN={avg_feat(fn, feat):.3f}  TP={avg_feat(tp, feat):.3f}  "
              f"FP={avg_feat(fp, feat):.3f}")

    # sample qualitative examples for the write-up
    examples = {
        "false_negative_examples": [
            {"sentence": r["sentence"], "types": r["hallucination_types"],
             "top_evidence": r["top_evidence"][:1], "score": r["pred_score"]}
            for r in sorted(fn, key=lambda r: -r["pred_score"])[:5]  # near-miss FNs
        ],
        "false_positive_examples": [
            {"sentence": r["sentence"], "top_evidence": r["top_evidence"][:1],
             "score": r["pred_score"]}
            for r in sorted(fp, key=lambda r: r["pred_score"])[:5]  # confident-but-wrong FPs... actually high score
        ],
        "true_positive_examples": [
            {"sentence": r["sentence"], "types": r["hallucination_types"],
             "top_evidence": r["top_evidence"][:1], "score": r["pred_score"]}
            for r in sorted(tp, key=lambda r: -r["pred_score"])[:5]
        ],
    }

    out = {
        "counts": {"fn": len(fn), "fp": len(fp), "tp": len(tp)},
        "recall_by_hallucination_type": type_recall,
        "avg_claim_len_words": {"fn": avg_len(fn), "tp": avg_len(tp), "fp": avg_len(fp)},
        "examples": examples,
    }
    with open(RESULTS_DIR / "error_analysis.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved error_analysis.json")


if __name__ == "__main__":
    main()
