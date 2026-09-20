import argparse
import csv
import json
import re
import random
from pathlib import Path
from typing import Any, cast

from llm_ling.client import chat_completion
from llm_ling.prompts import build_prompt


CONDITIONS = [
    "baseline",
    "self_classify",
    "correct_prime",
    "sham_prime_wrong",
    "sham_prime_irrelevant",
    "no_context",
]


def load_puzzles(path, limit=None):
    with open(path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if limit is not None:
        return rows[:limit]
    return rows


def find_sham_category(category):
    options = ["phonology", "morphology", "syntax", "semantics"]
    current = (category or "").lower()
    for candidate in options:
        if candidate != current:
            return candidate
    return "phonology"


def parse_response(text, finish_reason=None):
    info = {"status": "empty", "has_answer": False, "has_category": False}
    if text is None:
        if finish_reason == "length":
            info["status"] = "truncated"
        return info

    cleaned = text.strip()
    if not cleaned:
        if finish_reason == "length":
            info["status"] = "truncated"
        return info

    for key in ("JUSTIFICATION", "REASONING", "CATEGORY", "ANSWER"):
        match = re.search(rf"{key}:\s*(.*?)(?=\n[A-Z_]+:|$)", cleaned, flags=re.DOTALL)
        if match:
            info[key] = match.group(1).strip()

    if "ANSWER:" in cleaned:
        info["has_answer"] = True
    if "CATEGORY:" in cleaned:
        info["has_category"] = True

    if info.get("ANSWER"):
        info["status"] = "parsed"
    elif "JUSTIFICATION:" in cleaned or "REASONING:" in cleaned or finish_reason == "length":
        info["status"] = "truncated"
    else:
        info["status"] = "format_error"

    return info


def run_condition(puzzle, condition, model="qwen/qwen3-32b", max_tokens=4096):
    category = (puzzle.get("category") or puzzle.get("category_gold") or "phonology").strip()
    puzzle_text = puzzle["puzzle_text"]
    puzzle_id = puzzle.get("puzzle_id")

    if condition == "baseline":
        prompt = build_prompt("baseline", puzzle_text)
    elif condition == "self_classify":
        prompt = build_prompt("self_classify", puzzle_text)
    elif condition == "correct_prime":
        prompt = build_prompt("correct_prime", puzzle_text, category=category)
    elif condition == "sham_prime_wrong":
        prompt = build_prompt("sham_prime_wrong", puzzle_text, wrong_category=find_sham_category(category))
    elif condition == "sham_prime_irrelevant":
        prompt = build_prompt("sham_prime_irrelevant", puzzle_text)
    elif condition == "no_context":
        prompt = build_prompt("no_context", puzzle_text, puzzle_id=puzzle_id)
    else:
        raise ValueError(f"unknown condition: {condition}")

    completion = cast(dict[str, Any], chat_completion(
        prompt,
        model=model,
        max_tokens=max_tokens,
        return_metadata=True,
    ))
    response = completion["text"]
    parsed = parse_response(response, completion["finish_reason"])
    answer_terms = [term.strip().lower() for term in puzzle.get("answer_terms", "").split("|") if term.strip()]
    answer_text = parsed.get("ANSWER", "").lower()
    category_text = parsed.get("CATEGORY", "").lower()
    return {
        "puzzle_id": puzzle_id,
        "condition": condition,
        "category": category,
        "gold_answer": puzzle.get("gold_answer", ""),
        "category_match": category in category_text if parsed.get("has_category") else None,
        "answer_term_match": any(term in answer_text for term in answer_terms) if answer_terms else None,
        "model": completion["model"],
        "finish_reason": completion["finish_reason"],
        "usage": completion["usage"],
        "prompt": prompt,
        "response": response,
        "parsed": parsed,
    }


def main():
    parser = argparse.ArgumentParser(description="Run the LingOly linguistics-puzzle experiment.")
    parser.add_argument("--puzzle-file", default="data/lingoly_puzzles.csv")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--conditions", nargs="*", default=CONDITIONS)
    parser.add_argument("--output", default="full_experiment.jsonl")
    parser.add_argument("--resume", action="store_true", help="skip puzzle/condition pairs already in --output")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument("--model", default="qwen/qwen3-32b")
    parser.add_argument("--max-tokens", type=int, default=16384)
    args = parser.parse_args()

    puzzle_file = Path(args.puzzle_file)
    if not puzzle_file.exists():
        raise FileNotFoundError(f"{puzzle_file} is required before running the experiment")
    if args.repeats < 1:
        raise ValueError("--repeats must be positive")
    if args.max_tokens < 1:
        raise ValueError("--max-tokens must be positive")

    rows = load_puzzles(puzzle_file, limit=args.limit)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done = set()
    if args.resume and out_path.exists():
        with out_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    prev = json.loads(line)
                    if prev.get("status") != "error":
                        done.add((prev["puzzle_id"], prev["condition"], prev.get("trial", 0)))

    mode = "a" if args.resume else "w"
    rng = random.Random(args.seed)
    with out_path.open(mode, encoding="utf-8") as f:
        for puzzle in rows:
            for trial in range(args.repeats):
                conditions = list(args.conditions)
                rng.shuffle(conditions)
                for condition in conditions:
                    key = (puzzle.get("puzzle_id"), condition, trial)
                    if key in done:
                        continue
                    try:
                        result = run_condition(puzzle, condition, args.model, args.max_tokens)
                        result["trial"] = trial
                        result["seed"] = args.seed
                        result["model_requested"] = args.model
                    except Exception as exc:
                        result = {
                            "puzzle_id": puzzle.get("puzzle_id"),
                            "condition": condition,
                            "trial": trial,
                            "seed": args.seed,
                            "status": "error",
                            "error": str(exc),
                        }
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    f.flush()
                    response = result.get("response") or ""
                    print(json.dumps({
                        "puzzle_id": result["puzzle_id"],
                        "condition": condition,
                        "trial": trial,
                        "status": result.get("parsed", {}).get("status", "error"),
                        "response_preview": response[:220],
                    }, ensure_ascii=False))

    print(f"experiment complete: {len(rows)} puzzles x {len(args.conditions)} conditions x {args.repeats} trials -> {out_path}")


if __name__ == "__main__":
    main()
