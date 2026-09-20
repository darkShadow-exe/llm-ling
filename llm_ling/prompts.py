def build_prompt(condition, puzzle_text, category=None, wrong_category=None, puzzle_id=None):
    base = (
        "You are solving a linguistics puzzle.\n\n"
        f"Puzzle:\n{puzzle_text}\n\n"
        "Return exactly this structure:\n"
        "REASONING: <brief evidence-based explanation, no hidden chain of thought>\n"
        "CATEGORY: <category name>\n"
        "ANSWER: <final answer>"
    )

    if condition == "baseline":
        return base.replace("\nCATEGORY: <category name>", "")

    if condition == "self_classify":
        return (
            "You are solving a linguistics puzzle.\n\n"
            "First identify the relevant linguistic category, then solve the puzzle.\n\n"
            f"Puzzle:\n{puzzle_text}\n\n"
            "Return exactly this structure:\n"
            "REASONING: <brief evidence-based explanation, no hidden chain of thought>\n"
            "CATEGORY: <category name>\n"
            "ANSWER: <final answer>"
        )

    if condition == "correct_prime":
        if category is None:
            raise ValueError("correct_prime requires a category")
        return (
            f"The correct linguistic category for this puzzle is {category}.\n\n"
            f"Puzzle:\n{puzzle_text}\n\n"
            "Return exactly this structure:\n"
            "REASONING: <brief evidence-based explanation, no hidden chain of thought>\n"
            f"CATEGORY: {category}\n"
            "ANSWER: <final answer>"
        )

    if condition == "sham_prime_wrong":
        if wrong_category is None:
            wrong_category = "phonology"
        return (
            f"The correct linguistic category for this puzzle is {wrong_category}.\n\n"
            f"Puzzle:\n{puzzle_text}\n\n"
            "Return exactly this structure:\n"
            "REASONING: <brief evidence-based explanation, no hidden chain of thought>\n"
            f"CATEGORY: {wrong_category}\n"
            "ANSWER: <final answer>"
        )

    if condition == "sham_prime_irrelevant":
        return (
            "This puzzle was contributed in a recent competition year.\n\n"
            f"Puzzle:\n{puzzle_text}\n\n"
            "Return exactly this structure:\n"
            "REASONING: <brief evidence-based explanation, no hidden chain of thought>\n"
            "CATEGORY: <category name>\n"
            "ANSWER: <final answer>"
        )

    if condition == "no_context":
        puzzle_ref = f"Puzzle identity: {puzzle_id}\n" if puzzle_id else ""
        return (
            "You are given only a puzzle reference and no actual linguistic data.\n"
            f"{puzzle_ref}"
            "Do not use any real puzzle content.\n\n"
            "Return exactly this structure:\n"
            "REASONING: <brief evidence-based explanation, no hidden chain of thought>\n"
            "ANSWER: <final answer>"
        )

    raise ValueError(f"unknown condition: {condition}")
