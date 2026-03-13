"""
step2_ollama_preannotate_big.py

Reads approach_a/output/candidates_big.csv.
Writes approach_a/output/preannotated_big.csv.

For each candidate, asks Ollama (LLM) to return:
- register_label: one of
    "informal", "neutral essay", "academic", "news", "ambiguous"
- register_markedness_llm: 0 or 1  (1 = register-marked / too formal)
- substitution_natural_llm: 0 or 1 (1 = suggested alternative is natural)
- simpler_alternative_llm: a simpler English word or "" if none
- confidence_llm: integer 1, 2 or 3  (3 = highest confidence)
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


OUTPUT_DIR = Path(__file__).resolve().parent / "output"
CANDIDATES_CSV = OUTPUT_DIR / "candidates_big.csv"
PREANNOTATED_CSV = OUTPUT_DIR / "preannotated_big.csv"

OLLAMA_MODEL = "mistral:7b"
OLLAMA_URL = "http://localhost:11434"

# JSON schema for one item (we ask for a list of these)
ITEM_SCHEMA = {
    "type": "object",
    "properties": {
        "register_label": {
            "type": "string",
            "enum": ["informal", "neutral essay", "academic", "news", "ambiguous"],
        },
        "register_markedness_llm": {
            "type": "integer",
            "enum": [0, 1],
        },
        "substitution_natural_llm": {
            "type": "integer",
            "enum": [0, 1],
        },
        "simpler_alternative_llm": {
            "type": "string",
        },
        "confidence_llm": {
            "type": "integer",
            "enum": [1, 2, 3],
        },
    },
    "required": [
        "register_label",
        "register_markedness_llm",
        "substitution_natural_llm",
        "simpler_alternative_llm",
        "confidence_llm",
    ],
    "additionalProperties": False,
}

# Full response: object with "results" array
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "results": {
            "type": "array",
            "items": ITEM_SCHEMA,
        },
    },
    "required": ["results"],
    "additionalProperties": False,
}


def build_prompt(batch: List[Dict]) -> str:
    items = [
        {
            "sentence_id": int(row["sentence_id"]),
            "target_word": row["target_word"],
            "text": row["text"],
        }
        for row in batch
    ]
    user_payload = json.dumps(items, ensure_ascii=False, indent=2)
    return (
        "You are a careful linguist.\n"
        "For EACH item below, you see a learner sentence and a Latinate target_word.\n\n"
        "Your job is to return, for each item, the following fields:\n"
        "1) register_label: one of exactly these five strings:\n"
        '   - \"informal\"\n'
        '   - \"neutral essay\"  (typical school essay / learner writing)\n'
        '   - \"academic\"\n'
        '   - \"news\"\n'
        '   - \"ambiguous\" (if you really cannot decide)\n'
        "   This label describes the overall register of the CONTEXT, not just the word.\n\n"
        "2) register_markedness_llm: 0 or 1.\n"
        "   - 1 if the target_word is register-marked / too formal for THIS context.\n"
        "   - 0 if it is not register-marked in this context.\n\n"
        "3) substitution_natural_llm: 0 or 1.\n"
        "   - 1 if the simpler_alternative_llm you give would be a natural replacement\n"
        "     in THIS sentence.\n"
        "   - 0 if no simple, clearly better alternative comes to mind.\n\n"
        "4) simpler_alternative_llm: a single simpler English word that could\n"
        "   replace the target_word in this sentence, or an empty string \"\" if none.\n\n"
        "5) confidence_llm: 1, 2, or 3.\n"
        "   - 1 = low confidence\n"
        "   - 2 = medium confidence\n"
        "   - 3 = high confidence\n\n"
        "Return a JSON object with key \"results\": a list of objects, one per item,\n"
        "in the same order, each with EXACTLY these fields:\n"
        "  - register_label\n"
        "  - register_markedness_llm\n"
        "  - substitution_natural_llm\n"
        "  - simpler_alternative_llm\n"
        "  - confidence_llm\n\n"
        + user_payload
    )


def _extract_results_from_content(content: str) -> List[Dict]:
    """Parse JSON from model output; accept either {results: [...]} or bare [...]."""
    content = content.strip()
    if content.startswith("```"):
        lines = content.split("\n")
        if lines[0].strip().startswith("```"):
            content = "\n".join(lines[1:])
        content = content.replace("```", "").strip()
    start = content.find("{")
    if start == -1:
        start = content.find("[")
    if start == -1:
        raise ValueError("No JSON object or array in response")
    depth = 0
    opener = content[start]
    closer = "]" if opener == "[" else "}"
    raw = None
    for i in range(start, len(content)):
        if content[i] == opener:
            depth += 1
        elif content[i] == closer:
            depth -= 1
            if depth == 0:
                raw = content[start : i + 1]
                break
    if raw is None:
        raise ValueError("Unclosed JSON in response")
    data = json.loads(raw)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "results" in data:
        return data["results"]
    raise ValueError(f"Expected list or {{results: list}}, got {type(data)}")


def _request_ollama(body: dict) -> str:
    """
    Try Ollama's /api/chat first; if not available (404), fall back to /api/generate.
    Returns the model text content (string) containing JSON.
    """

    def _post(path: str, payload: dict) -> dict:
        req = urllib.request.Request(
            f"{OLLAMA_URL}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=300) as resp:
            return json.loads(resp.read().decode("utf-8"))

    # /api/chat (newer Ollama)
    try:
        out = _post("/api/chat", body)
        content = out.get("message", {}).get("content", "")
        if not content or not content.strip():
            raise ValueError("Ollama /api/chat returned empty content")
        return content
    except urllib.error.HTTPError as e:
        if e.code != 404:
            raise

    # /api/generate (older Ollama)
    gen_body = {
        "model": body.get("model"),
        "prompt": body.get("messages", [{}])[0].get("content", ""),
        "stream": False,
        "options": body.get("options", {}),
        "format": body.get("format", "json"),
    }
    out = _post("/api/generate", gen_body)
    content = out.get("response", "")
    if not content or not content.strip():
        raise ValueError("Ollama /api/generate returned empty content")
    return content


def call_ollama(prompt: str) -> List[Dict]:
    body = {
        "model": OLLAMA_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": 0},
    }
    # Try with schema first (Ollama late 2024+)
    body["format"] = RESPONSE_SCHEMA
    try:
        content = _request_ollama(body)
        data = json.loads(content)
        if isinstance(data, dict) and "results" in data and isinstance(
            data["results"], list
        ):
            return data["results"]
    except (ValueError, json.JSONDecodeError, OSError):
        pass
    # Fallback: plain JSON, parse flexibly
    body["format"] = "json"
    content = _request_ollama(body)
    return _extract_results_from_content(content)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(CANDIDATES_CSV)
    print(f"Loaded {df.shape[0]} rows from {CANDIDATES_CSV}")
    records = df.to_dict(orient="records")
    all_results: Dict[Tuple[int, str], Dict] = {}

    BATCH_SIZE = 5

    for start in range(0, len(records), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(records))
        batch = records[start:end]
        print(f"Ollama rows {start}–{end - 1}...")
        prompt = build_prompt(batch)
        n = len(batch)

        try:
            batch_results = call_ollama(prompt)
        except (ValueError, json.JSONDecodeError, OSError) as e:
            print(f"  Warning: {e}; using defaults for this batch.")
            batch_results = []

        if len(batch_results) > n:
            batch_results = batch_results[:n]
        while len(batch_results) < n:
            batch_results.append(
                {
                    "register_label": "ambiguous",
                    "register_markedness_llm": 0,
                    "substitution_natural_llm": 0,
                    "simpler_alternative_llm": "",
                    "confidence_llm": 1,
                }
            )

        for i, item in enumerate(batch_results):
            sid = int(batch[i]["sentence_id"])
            tw = batch[i]["target_word"]

            label = str(item.get("register_label", "ambiguous")).strip()
            if label not in ["informal", "neutral essay", "academic", "news", "ambiguous"]:
                label = "ambiguous"

            reg_mark = int(item.get("register_markedness_llm", 0))
            subst_nat = int(item.get("substitution_natural_llm", 0))

            alt = str(item.get("simpler_alternative_llm", "")).strip()
            if len(alt) > 50:
                alt = alt[:50]

            try:
                conf = int(item.get("confidence_llm", 1))
            except (TypeError, ValueError):
                conf = 1
            if conf not in (1, 2, 3):
                conf = 1

            all_results[(sid, tw)] = {
                "register_label": label,
                "register_markedness_llm": reg_mark,
                "substitution_natural_llm": subst_nat,
                "simpler_alternative_llm": alt,
                "confidence_llm": conf,
            }

    df_out = df.copy()

    df_out["register_label"] = [
        all_results.get((int(row["sentence_id"]), row["target_word"]), {}).get(
            "register_label", ""
        )
        for _, row in df.iterrows()
    ]
    df_out["register_markedness_llm"] = [
        all_results.get((int(row["sentence_id"]), row["target_word"]), {}).get(
            "register_markedness_llm", ""
        )
        for _, row in df.iterrows()
    ]
    df_out["substitution_natural_llm"] = [
        all_results.get((int(row["sentence_id"]), row["target_word"]), {}).get(
            "substitution_natural_llm", ""
        )
        for _, row in df.iterrows()
    ]
    df_out["simpler_alternative_llm"] = [
        all_results.get((int(row["sentence_id"]), row["target_word"]), {}).get(
            "simpler_alternative_llm", ""
        )
        for _, row in df.iterrows()
    ]
    df_out["confidence_llm"] = [
        all_results.get((int(row["sentence_id"]), row["target_word"]), {}).get(
            "confidence_llm", ""
        )
        for _, row in df.iterrows()
    ]

    df_out.to_csv(PREANNOTATED_CSV, index=False)
    print(f"Wrote {PREANNOTATED_CSV}")


if __name__ == "__main__":
    main()

