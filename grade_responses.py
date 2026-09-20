import argparse
import csv
import hashlib
import json
import random
import statistics
from pathlib import Path

import requests

from llm_ling.client import chat_completion, load_env_file


DEFAULT_MODELS = ["qwen/qwen3-32b"]
ERROR_LABELS = (
    "wrong_answer",
    "incomplete_answer",
    "wrong_rule",
    "unsupported_claim",
    "misread_data",
    "format_issue",
)


RUBRIC = """Grade only the response, not its condition or generating model.
Answer score: 0 = wrong, 1 = partly correct or incomplete, 2 = fully correct.
Reasoning score: 0 = wrong or unsupported; 1 = useful observation but a substantive gap;
2 = correct answer supported by relevant evidence; 3 = correct, complete, and tightly tied
to the decisive data.
Evidence score: 0 = does not use the puzzle data; 1 = uses some relevant data;
2 = cites the decisive contrast or derivation.
Return JSON only with integer scores, error_labels, confidence from 0 to 1, and a short rationale.
"""


def normalize(text):
    return " ".join((text or "").lower().strip().split())


def neutral_puzzle(prompt):
    text = prompt or ""
    if "\n\nPuzzle:\n" in text:
        text = text.split("\n\nPuzzle:\n", 1)[1]
    if "\n\nReturn exactly this structure:" in text:
        text = text.split("\n\nReturn exactly this structure:", 1)[0]
    return text.strip()


def make_packet(row, packet_id):
    parsed = row.get("parsed", {})
    return {
        "packet_id": packet_id,
        "puzzle_id": row.get("puzzle_id"),
        "condition": row.get("condition"),
        "trial": row.get("trial", 0),
        "category": row.get("category", ""),
        "model": row.get("model", ""),
        "puzzle": neutral_puzzle(row.get("prompt", "")),
        "gold_answer": row.get("gold_answer", ""),
        "answer": parsed.get("ANSWER", ""),
        "reasoning": parsed.get("REASONING", ""),
    }


def load_references(path):
    if not path:
        return {}
    with open(path, encoding="utf-8") as f:
        return {
            row["puzzle_id"]: row.get("solution", "")
            for row in (json.loads(line) for line in f if line.strip())
        }


def search_reference(puzzle, token, count=5, endpoint="hackclub"):
    query = " ".join(puzzle["puzzle"].split())[:500]
    query = f"{query} linguistics olympiad solution explanation"
    if endpoint == "brave":
        url = "https://api.search.brave.com/res/v1/web/search"
        headers = {"X-Subscription-Token": token}
    else:
        url = "https://search.hackclub.com/res/v1/web/search"
        headers = {"x-subscription-token": token}
    response = requests.get(
        url,
        params={"q": query, "count": count},
        headers=headers,
        timeout=60,
    )
    response.raise_for_status()
    data = response.json()
    results = data.get("web", {}).get("results", [])
    return "\n".join(
        f"Title: {item.get('title', '')}\nURL: {item.get('url', '')}\nSnippet: {item.get('description', '')}"
        for item in results[:count]
    )


def judge_prompt(packet):
    solution = packet.get("extended_solution", "")
    solution_block = ""
    if solution:
        solution_block = f"""
EXTENDED REFERENCE SOLUTION:
Use this only to evaluate the response. Do not assume the answer is correct merely because it resembles this reference.
{solution}
"""
    return f"""You are an independent evaluator for a linguistics reasoning benchmark.
Do not infer or discuss the experimental condition. Grade the response against the puzzle and gold answer.

{RUBRIC}

PUZZLE AND REQUIRED OUTPUT:
{packet['puzzle']}

REFERENCE ANSWER:
{packet['gold_answer']}
{solution_block}

MODEL ANSWER:
{packet['answer']}

MODEL JUSTIFICATION:
{packet['reasoning']}

Allowed error labels: {', '.join(ERROR_LABELS)}.
Return exactly one JSON object with keys:
answer_score, reasoning_score, evidence_score, error_labels, confidence, rationale.
"""


def parse_judgment(text):
    cleaned = (text or "").strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("judge returned no JSON object")
    result = json.loads(cleaned[start:end + 1])
    for key in ("answer_score", "reasoning_score", "evidence_score"):
        value = int(result[key])
        if key == "reasoning_score" and value not in range(4):
            raise ValueError(f"invalid {key}: {value}")
        if key != "reasoning_score" and value not in range(3):
            raise ValueError(f"invalid {key}: {value}")
        result[key] = value
    result["error_labels"] = [label for label in result.get("error_labels", []) if label in ERROR_LABELS]
    result["confidence"] = max(0.0, min(1.0, float(result.get("confidence", 0))))
    result["rationale"] = str(result.get("rationale", ""))
    return result


def aggregate(judgments):
    if not judgments:
        return {"judge_count": 0, "disagreement": True}
    scores = {}
    for key in ("answer_score", "reasoning_score", "evidence_score"):
        values = [item[key] for item in judgments]
        scores[key + "_median"] = statistics.median(values)
        scores[key + "_range"] = max(values) - min(values)
    labels = sorted({label for item in judgments for label in item["error_labels"]})
    scores["error_labels"] = labels
    scores["judge_count"] = len(judgments)
    scores["disagreement"] = any(scores[key + "_range"] > 1 for key in ("answer_score", "reasoning_score", "evidence_score"))
    return scores


def review_row(packet, seed):
    digest = hashlib.sha256(f"{seed}:{packet['packet_id']}".encode()).hexdigest()[:12]
    return {
        "review_id": digest,
        "puzzle_id": packet["puzzle_id"],
        "answer": packet["answer"],
        "reasoning": packet["reasoning"],
        "rater_1_answer_score": "",
        "rater_1_reasoning_score": "",
        "rater_1_evidence_score": "",
        "rater_2_answer_score": "",
        "rater_2_reasoning_score": "",
        "rater_2_evidence_score": "",
        "notes": "",
    }


def load_rows(path):
    with open(path, encoding="utf-8") as f:
        return [
            row for row in (json.loads(line) for line in f if line.strip())
            if row.get("status") != "error"
        ]


def main():
    parser = argparse.ArgumentParser(description="Run blinded LLM-council grading.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="council_grades.jsonl")
    parser.add_argument("--review-output", help="write an optional blinded human-review CSV")
    parser.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    parser.add_argument("--judges", type=int, default=3)
    parser.add_argument("--review-fraction", type=float, default=0.2)
    parser.add_argument("--reference-solutions", help="JSONL with puzzle_id and solution, visible only to judges")
    parser.add_argument("--web-search", action="store_true", help="retrieve search references for judges only")
    parser.add_argument("--search-count", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.judges < 1:
        raise ValueError("--judges must be positive")
    if not 0 <= args.review_fraction <= 1:
        raise ValueError("--review-fraction must be between 0 and 1")
    if args.search_count < 1:
        raise ValueError("--search-count must be positive")

    rows = load_rows(args.input)
    references = load_references(args.reference_solutions)
    search_token = None
    search_endpoint = "hackclub"
    if args.web_search:
        env = load_env_file()
        if env.get("HACK_CLUB_SEARCH_API_KEY"):
            search_token = env["HACK_CLUB_SEARCH_API_KEY"]
        elif env.get("BRAVE_API_KEY"):
            search_token = env["BRAVE_API_KEY"]
            search_endpoint = "brave"
        else:
            search_token = env.get("HACK_CLUB_AI_API_KEY") or env.get("HACKLCUB_API_KEY")
        if not search_token:
            raise RuntimeError("--web-search requires a Hack Club Search/AI key or BRAVE_API_KEY in .env")
    rng = random.Random(args.seed)
    packets = []
    for index, row in enumerate(rows):
        packet_id = hashlib.sha256(f"{args.seed}:{index}:{row.get('puzzle_id')}".encode()).hexdigest()[:16]
        packet = make_packet(row, packet_id)
        packet["extended_solution"] = references.get(packet["puzzle_id"], "")
        if search_token and not args.dry_run:
            packet["extended_solution"] = search_reference(packet, search_token, args.search_count, search_endpoint)
        packets.append(packet)

    review_count = round(len(packets) * args.review_fraction) if args.review_output else 0
    review_ids = {packet["packet_id"] for packet in rng.sample(packets, review_count)} if review_count else set()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.review_output:
        review_path = Path(args.review_output)
        review_path.parent.mkdir(parents=True, exist_ok=True)
        with review_path.open("w", newline="", encoding="utf-8") as f:
            fields = list(review_row(packets[0], args.seed).keys()) if packets else []
            writer = csv.DictWriter(f, fieldnames=fields)
            if fields:
                writer.writeheader()
                for packet in packets:
                    if packet["packet_id"] in review_ids:
                        writer.writerow(review_row(packet, args.seed))

    with output_path.open("w", encoding="utf-8") as f:
        for packet in packets:
            judgments = []
            assigned = list(args.models)
            rng.shuffle(assigned)
            assigned = [assigned[index % len(assigned)] for index in range(args.judges)]
            if not args.dry_run:
                for model in assigned:
                    completion = chat_completion(judge_prompt(packet), model=model, temperature=0.0, return_metadata=True)
                    judgment = parse_judgment(completion["text"])
                    judgment["judge_model"] = model
                    judgment["finish_reason"] = completion["finish_reason"]
                    judgments.append(judgment)
            result = {
                "packet_id": packet["packet_id"],
                "puzzle_id": packet["puzzle_id"],
                "condition": packet["condition"],
                "trial": packet["trial"],
                "category": packet["category"],
                "model": packet["model"],
                "judgments": judgments,
                "aggregate": aggregate(judgments),
                "review_selected": packet["packet_id"] in review_ids,
            }
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"graded {len(packets)} responses; human review rows: {len(review_ids)}")


if __name__ == "__main__":
    main()
