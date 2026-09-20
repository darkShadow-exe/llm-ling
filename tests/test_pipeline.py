import pytest

from llm_ling.prompts import build_prompt
from grade_responses import aggregate, judge_prompt, load_rows, neutral_puzzle, parse_judgment, search_reference
from analyze_results import mcnemar, holm
from run_experiment import parse_response


def test_prompt_schema_is_consistent_across_conditions():
    prompt = build_prompt("baseline", "data", category="phonology")
    assert "REASONING:" in prompt
    assert "ANSWER:" in prompt

    prompt2 = build_prompt("correct_prime", "data", category="syntax")
    assert "CATEGORY: syntax" in prompt2

    prompt3 = build_prompt("no_context", "data", puzzle_id="UKLO-2019-5")
    assert "Puzzle identity" in prompt3
    assert "ANSWER:" in prompt3


def test_parse_response_flags_truncation():
    text = "REASONING: I started analyzing the pattern but the model stopped early."
    parsed = parse_response(text)
    assert parsed["status"] == "truncated"
    assert parsed["has_answer"] is False

    text2 = "REASONING: I examined the data.\nCATEGORY: syntax"
    parsed2 = parse_response(text2)
    assert parsed2["status"] == "truncated"
    assert parsed2["has_category"] is True

    text3 = "REASONING: I examined the data.\nCATEGORY: syntax\nANSWER: VSO order."
    parsed3 = parse_response(text3)
    assert parsed3["status"] == "parsed"
    assert parsed3["has_answer"] is True

    parsed4 = parse_response("", finish_reason="length")
    assert parsed4["status"] == "truncated"

    parsed5 = parse_response("JUSTIFICATION: The suffix recurs in both plural forms.")
    assert parsed5["status"] == "truncated"


def test_parse_judgment_accepts_json_wrapped_in_text():
    judgment = parse_judgment(
        'Here is the grade: {"answer_score": 2, "reasoning_score": 3, '
        '"evidence_score": 2, "error_labels": [], "confidence": 0.9, '
        '"rationale": "Correct rule and evidence."}'
    )
    assert judgment["answer_score"] == 2
    assert judgment["reasoning_score"] == 3
    assert judgment["confidence"] == 0.9


def test_neutral_puzzle_removes_condition_wrapper():
    prompt = "The correct linguistic category for this puzzle is syntax.\n\nPuzzle:\ndata\n\nReturn exactly this structure:\nANSWER: <final answer>"
    assert neutral_puzzle(prompt) == "data"


def test_extended_solution_is_only_in_judge_prompt():
    prompt = judge_prompt({
        "puzzle": "data",
        "gold_answer": "answer",
        "answer": "model answer",
        "reasoning": "model reasoning",
        "extended_solution": "worked derivation",
    })
    assert "worked derivation" in prompt
    assert "model answer" in prompt


def test_search_reference_formats_web_results(monkeypatch):
    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"web": {"results": [{"title": "Solution", "url": "https://example.test", "description": "Worked derivation"}]}}

    monkeypatch.setattr("grade_responses.requests.get", lambda *args, **kwargs: Response())
    result = search_reference({"puzzle": "linguistics puzzle"}, "token", count=1)
    assert "Worked derivation" in result
    assert "https://example.test" in result


def test_search_reference_supports_brave_endpoint(monkeypatch):
    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"web": {"results": []}}

    calls = []
    monkeypatch.setattr("grade_responses.requests.get", lambda *args, **kwargs: calls.append((args, kwargs)) or Response())
    search_reference({"puzzle": "linguistics puzzle"}, "token", count=1, endpoint="brave")
    assert calls[0][0][0] == "https://api.search.brave.com/res/v1/web/search"
    assert calls[0][1]["headers"] == {"X-Subscription-Token": "token"}


def test_aggregate_uses_medians_and_flags_disagreement():
    judgments = [
        {"answer_score": 2, "reasoning_score": 3, "evidence_score": 2, "error_labels": [], "confidence": 1, "rationale": ""},
        {"answer_score": 1, "reasoning_score": 2, "evidence_score": 1, "error_labels": ["incomplete_answer"], "confidence": 0.7, "rationale": ""},
        {"answer_score": 0, "reasoning_score": 0, "evidence_score": 0, "error_labels": ["wrong_answer"], "confidence": 0.8, "rationale": ""},
    ]
    result = aggregate(judgments)
    assert result["answer_score_median"] == 1
    assert result["reasoning_score_median"] == 2
    assert result["disagreement"] is True
    assert result["error_labels"] == ["incomplete_answer", "wrong_answer"]


def test_mcnemar_and_holm():
    result = mcnemar([1, 1, 0, 0], [1, 0, 1, 0])
    assert result["n"] == 4
    assert result["discordant_left_only"] == 1
    assert result["discordant_right_only"] == 1
    assert holm([0.01, 0.04]) == [0.02, 0.04]


def test_grader_ignores_failed_model_calls(tmp_path):
    path = tmp_path / "results.jsonl"
    path.write_text(
        '{"puzzle_id":"ok","condition":"baseline"}\n'
        '{"puzzle_id":"failed","condition":"baseline","status":"error"}\n',
        encoding="utf-8",
    )
    assert [row["puzzle_id"] for row in load_rows(path)] == ["ok"]
