import json
import numpy as np
import argparse

"""
This script parses the raw results from the faithfulness evaluation and computes the top-k faithfulness score i.e., FAITH@k
"""

# The computation is intuitive: ratio of "visible" concepts in the top-k concepts.
def compute_topk(results, max_k=5):
    topk_scores = {k: [] for k in range(1, max_k + 1)}

    for r in results:
        top5 = r["top5"]

        for k in range(1, max_k + 1):
            topk = top5[:k]
            visible_count = sum(1 for c in topk if c["visible"])
            score = visible_count / k
            topk_scores[k].append(score)

    return {k: np.mean(v) for k, v in topk_scores.items()}


def compute_topk_split(results, max_k=5):
    correct = [r for r in results if r["correct"]]
    incorrect = [r for r in results if not r["correct"]]

    return {
        "all": compute_topk(results, max_k),
        "correct": compute_topk(correct, max_k),
        "incorrect": compute_topk(incorrect, max_k) if len(incorrect) > 0 else {}
    }


def main(args):
    with open(args.input_json) as f:
        results = json.load(f)

    stats = compute_topk_split(results, max_k=5)

    print("\n=== FAITH@k ===")

    for split in ["all"]: #, "correct", "incorrect"]:
        print(f"\n--- {split.upper()} ---")
        if split not in stats or len(stats[split]) == 0:
            print("No data")
            continue

        for k in range(1, 6):
            val = stats[split].get(k, None)
            if val is not None:
                print(f"Top-{k}: {val:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", required=True)
    args = parser.parse_args()

    main(args)