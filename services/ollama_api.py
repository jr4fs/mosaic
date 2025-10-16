# services/ollama_api.py
"""
Wrapper functions for Ollama interactions.
Keeps network/CLI logic in one place so the UI modules stay tidy.
"""

import subprocess
import requests
import json
from typing import List, Generator

OLLAMA_URL = "http://localhost:11434"


def get_installed_models() -> List[str]:
    """
    Use `ollama list` to detect installed models. Return a fallback list on failure.
    """
    try:
        proc = subprocess.run(["ollama", "list"], capture_output=True, text=True, check=True)
        lines = proc.stdout.strip().splitlines()
        models = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if line.lower().startswith("name") and "id" in line.lower():
                continue
            parts = line.split()
            if parts:
                models.append(parts[0])
        # dedupe preserving order
        seen = set()
        uniq = [m for m in models if not (m in seen or seen.add(m))]
        return uniq
    except Exception:
        # fallback if ollama CLI not available or fails
        return ["llama3:latest", "llama3.2:latest", "mistral:latest"]


def generate_stream(model: str, prompt: str, timeout: int = 600) -> Generator[str, None, None]:
    """
    Stream text chunks from Ollama's /api/generate endpoint.
    Yields string chunks as they arrive.
    """
    url = f"{OLLAMA_URL}/api/generate"
    payload = {"model": model, "prompt": prompt, "stream": True}
    headers = {"Content-Type": "application/json"}
    with requests.post(url, json=payload, headers=headers, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        for line in r.iter_lines(decode_unicode=True):
            if not line:
                continue
            try:
                chunk = json.loads(line)
            except Exception:
                yield line
                continue
            # common shapes — adapt if your Ollama version uses others
            text = (
                chunk.get("response")
                or (chunk.get("message") or {}).get("content")
                or (chunk.get("choices") or [{}])[0].get("text")
            )
            if text is None:
                yield json.dumps(chunk)
            else:
                yield text


def generate_sync(model: str, prompt: str, timeout: int = 120) -> str:
    """
    Single-request (non-streaming) generate; returns text result or raw JSON string.
    """
    url = f"{OLLAMA_URL}/api/generate"
    payload = {"model": model, "prompt": prompt, "stream": False}
    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    j = r.json()
    return j.get("response") or (j.get("message") or {}).get("content") or (j.get("choices") or [{}])[0].get("text") or json.dumps(j)
