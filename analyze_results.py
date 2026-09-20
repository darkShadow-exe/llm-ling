import argparse
import json
from collections import defaultdict
from pathlib import Path

from scipy.stats import binomtest, chi2_contingency


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def council_score(row, threshold):
    aggregate = row.get("aggregate", {})
    score = aggregate.get("answer_score_median")
    return None if score is None else int(score >= threshold)


def mcnemar(left, right):
    pairs = [(a, b) for a, b in zip(left, right) if a is not None and b is not None]
    b = sum(a == 1 and c == 0 for a, c in pairs)
    c = sum(a == 0 and c == 1 for a, c in pairs)
    discordant = b + c
    p_value = 1.0 if discordant == 0 else binomtest(min(b, c), discordant, 0.5).pvalue
    odds_ratio = (b + 0.5) / (c + 0.5)
    return {
        "n": len(pairs),
        "left_accuracy": sum(a for a, _ in pairs) / len(pairs) if pairs else None,
        "right_accuracy": sum(c for _, c in pairs) / len(pairs) if pairs else None,
        "discordant_left_only": b,
        "discordant_right_only": c,
        "accuracy_difference": (sum(c for _, c in pairs) - sum(a for a, _ in pairs)) / len(pairs) if pairs else None,
        "odds_ratio_left_over_right": odds_ratio,
        "p_value": p_value,
    }


def holm(values):
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    adjusted = [1.0] * len(values)
    running = 0.0
    for rank, (index, value) in enumerate(ordered):
        corrected = min(1.0, (len(values) - rank) * value)
        running = max(running, corrected)
        adjusted[index] = running
    return adjusted


def pair_rows(rows, left_condition, right_condition, threshold):
    grouped = defaultdict(dict)
    for row in rows:
        key = (row.get("puzzle_id"), row.get("trial", 0))
        if row.get("condition") in (left_condition, right_condition):
            grouped[key][row["condition"]] = council_score(row, threshold)
    pairs = [values for values in grouped.values() if left_condition in values and right_condition in values]
    return [values[left_condition] for values in pairs], [values[right_condition] for values in pairs]


def classify_vs_solve(raw_rows, graded_rows, threshold):
    graded = {(row.get("puzzle_id"), row.get("condition"), row.get("trial", 0)): council_score(row, threshold) for row in graded_rows}
    table = [[0, 0], [0, 0]]
    for row in raw_rows:
        if row.get("condition") != "self_classify":
            continue
        category = row.get("category_match")
        answer = graded.get((row.get("puzzle_id"), row.get("condition"), row.get("trial", 0)))
        if category is None or answer is None:
            continue
        table[int(bool(category))][answer] += 1
    if sum(map(sum, table)) == 0:
        return {"n": 0}
    chi2, p_value, _, _ = chi2_contingency(table, correction=False)
    return {"n": sum(map(sum, table)), "table": table, "chi2": chi2, "p_value": p_value}


def main():
    parser = argparse.ArgumentParser(description="Analyze council-graded LingOly experiment results.")
    parser.add_argument("--raw", required=True, help="raw experiment JSONL")
    parser.add_argument("--graded", required=True, help="council output JSONL")
    parser.add_argument("--output", default="analysis.json")
    parser.add_argument("--threshold", type=int, default=2, help="minimum council answer score counted as correct")
    args = parser.parse_args()

    raw_rows = load_jsonl(args.raw)
    graded_rows = load_jsonl(args.graded)
    if args.threshold not in (1, 2):
        raise ValueError("--threshold must be 1 or 2")

    results = {"threshold": args.threshold, "graded_rows": len(graded_rows), "comparisons": {}}
    comparisons = [("baseline", "correct_prime"), ("correct_prime", "sham_prime_wrong"), ("correct_prime", "sham_prime_irrelevant")]
    p_values = []
    for left, right in comparisons:
        left_scores, right_scores = pair_rows(graded_rows, left, right, args.threshold)
        result = mcnemar(left_scores, right_scores)
        results["comparisons"][f"{left}_vs_{right}"] = result
        p_values.append(result["p_value"])
    adjusted = holm(p_values)
    for name, corrected in zip(results["comparisons"], adjusted):
        results["comparisons"][name]["holm_p_value"] = corrected

    results["self_classification"] = classify_vs_solve(raw_rows, graded_rows, args.threshold)
    results["category_comparisons"] = {}
    for category in sorted({row.get("category") for row in raw_rows if row.get("category")}):
        category_ids = {row.get("puzzle_id") for row in raw_rows if row.get("category") == category}
        category_graded = [row for row in graded_rows if row.get("puzzle_id") in category_ids]
        left_scores, right_scores = pair_rows(category_graded, "baseline", "correct_prime", args.threshold)
        results["category_comparisons"][category] = mcnemar(left_scores, right_scores)

    by_key = {(row.get("puzzle_id"), row.get("trial", 0), row.get("condition")): council_score(row, args.threshold) for row in graded_rows}
    adjusted = []
    for key, score in by_key.items():
        puzzle_id, trial, condition = key
        if condition == "no_context" or score is None:
            continue
        control = by_key.get((puzzle_id, trial, "no_context"))
        if control is not None:
            adjusted.append(score - control)
    results["contamination_adjustment"] = {
        "n": len(adjusted),
        "mean_delta": sum(adjusted) / len(adjusted) if adjusted else None,
    }

    Path(args.output).write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()