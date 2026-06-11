"""Minimal API tests for Groq and Google Gemini.

This script performs a single deterministic test call to each provider
using model names from the pipeline. It prints a JSON report with status
and short messages. It never prints API keys.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]


def load_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def test_groq(model_name: str) -> Dict[str, Any]:
    result: Dict[str, Any] = {"provider": "groq", "model": model_name}
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        result.update({"ok": False, "error": "GROQ_API_KEY not set"})
        return result

    try:
        from groq import Groq

        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": "Say ok."}],
            temperature=0.0,
            max_tokens=16,
        )
        text = getattr(response.choices[0].message, "content", "") or str(response)
        result.update({"ok": True, "message": text[:300]})
    except Exception as exc:
        result.update({"ok": False, "error": repr(exc)})
    return result


def test_gemini(model_name: str) -> Dict[str, Any]:
    result: Dict[str, Any] = {"provider": "gemini", "model": model_name}
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        result.update({"ok": False, "error": "GOOGLE_API_KEY not set"})
        return result
    try:
        import requests

        # List available models and pick a candidate that looks public/free.
        use_auth_header = isinstance(api_key, str) and (api_key.startswith('AQ.') or api_key.startswith('ya29.'))
        if use_auth_header:
            list_url = "https://generativelanguage.googleapis.com/v1beta2/models"
            lm = requests.get(list_url, headers={"Authorization": f"Bearer {api_key}"}, timeout=10).json()
        else:
            list_url = f"https://generativelanguage.googleapis.com/v1beta2/models?key={api_key}"
            lm = requests.get(list_url, timeout=10).json()
        models = [m.get("name") for m in lm.get("models", []) if m.get("name")]
        # heuristic candidates: prefer requested name, then discovered models, then common public names
        fallback_candidates = [
            "models/text-bison-001",
            "models/chat-bison-001",
            "text-bison-001",
            "chat-bison",
        ]
        candidates = [model_name] + models + fallback_candidates
        candidates = [c for i, c in enumerate(candidates) if c and c not in candidates[:i]]

        last_exc = None
        for candidate in candidates:
            # Try both full resource name and short name forms to avoid 404s.
            variants = [candidate]
            if not candidate.startswith("models/"):
                variants.insert(0, f"models/{candidate}")

            success_resp = None
            for var in variants:
                try:
                    if use_auth_header:
                        gen_url = f"https://generativelanguage.googleapis.com/v1beta2/{var}:generateText"
                        headers = {"Authorization": f"Bearer {api_key}"}
                        payload = {"prompt": {"text": "Say ok."}, "temperature": 0.0, "maxOutputTokens": 64}
                        resp = requests.post(gen_url, json=payload, headers=headers, timeout=15)
                    else:
                        gen_url = f"https://generativelanguage.googleapis.com/v1beta2/{var}:generateText?key={api_key}"
                        payload = {"prompt": {"text": "Say ok."}, "temperature": 0.0, "maxOutputTokens": 64}
                        resp = requests.post(gen_url, json=payload, timeout=15)
                    resp.raise_for_status()
                    success_resp = resp
                    success_var = var
                    break
                except Exception as exc:
                    last_exc = exc
                    continue

            if success_resp is None:
                continue

            try:
                data = success_resp.json()
                # Try several common locations for text
                text = None
                if isinstance(data, dict):
                    if "candidates" in data and data["candidates"]:
                        text = data["candidates"][0].get("output") or data["candidates"][0].get("content")
                    text = text or data.get("text") or data.get("output")
                text = (text or "").strip()
                result.update({"ok": True, "message": f"{success_var}: {text[:300]}"})
                last_exc = None
                break
            except Exception as exc:
                last_exc = exc
                continue

        if last_exc is not None and not result.get("ok"):
            raise last_exc
    except Exception as exc:
        result.update({"ok": False, "error": repr(exc)})
    return result


def main() -> int:
    # Load .env from project root (usual) and as a fallback from current
    # working directory (useful when running from other shells/IDE tasks).
    load_env_file(ROOT / ".env")
    load_env_file(Path.cwd() / ".env")
    # Default test models (same as pipeline)
    groq_model = "llama-3.3-70b-versatile"
    gemini_model = "gemini-1.5-flash"

    out = {"tests": []}

    out["tests"].append(test_groq(groq_model))
    out["tests"].append(test_gemini(gemini_model))

    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
