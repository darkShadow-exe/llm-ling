import json
import os
import time

import requests


def load_env_file(path=".env"):
    env = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                env[key.strip()] = value.strip().strip('"').strip("'")
    except FileNotFoundError:
        return {}
    return env


def get_api_key():
    env = load_env_file()
    for key_name in ("HACK_CLUB_AI_API_KEY", "HACKLCUB_API_KEY"):
        if env.get(key_name):
            return env[key_name]
    if os.getenv("HACK_CLUB_AI_API_KEY"):
        return os.getenv("HACK_CLUB_AI_API_KEY")
    if os.getenv("HACKLCUB_API_KEY"):
        return os.getenv("HACKLCUB_API_KEY")
    raise RuntimeError("missing Hack Club API key; set HACK_CLUB_AI_API_KEY or HACKLCUB_API_KEY")


def chat_completion(prompt, model="qwen/qwen3-32b", temperature=0.2, max_tokens=None, return_metadata=False):
    api_key = get_api_key()
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens

    # proxy occasionally returns a flaky non-JSON/5xx response, retry a couple times
    last_err = None
    for attempt in range(3):
        try:
            resp = requests.post(
                "https://ai.hackclub.com/proxy/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
            break
        except (requests.exceptions.RequestException, ValueError) as exc:
            last_err = exc
            if attempt < 2:
                time.sleep(2 * (attempt + 1))
    else:
        raise RuntimeError(f"Hack Club API request failed after retries: {last_err}")
    msg = data["choices"][0]["message"]
    if msg.get("content") is not None:
        text = msg["content"]
    elif msg.get("reasoning"):
        reasoning = msg["reasoning"].strip()
        if reasoning.startswith("REASONING:"):
            text = reasoning
        else:
            text = f"REASONING: {reasoning}"
    else:
        text = ""

    if return_metadata:
        return {
            "text": text,
            "model": data.get("model", model),
            "finish_reason": data["choices"][0].get("finish_reason"),
            "usage": data.get("usage"),
        }
    return text
