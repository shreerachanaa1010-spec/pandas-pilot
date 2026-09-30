"""Minimal REST adapters for Gemini and local Ollama teachers."""

from __future__ import annotations

import json
import os
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class Teacher(Protocol):
    """A provider that returns raw text for a generation prompt."""

    name: str

    def generate(self, prompt: str) -> str: ...


class TeacherError(RuntimeError):
    """Raised when a teacher cannot return a usable response."""


def _post_json(
    url: str, payload: dict[str, Any], headers: dict[str, str], timeout: float
) -> dict[str, Any]:
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read())
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as error:
        raise TeacherError(f"teacher request failed: {type(error).__name__}") from error
    if not isinstance(result, dict):
        raise TeacherError("teacher returned a non-object response")
    return result


class OllamaTeacher:
    """Call a local Ollama model through its JSON generation endpoint."""

    name = "ollama"

    def __init__(
        self,
        model: str = "qwen2.5-coder:7b",
        endpoint: str = "http://localhost:11434/api/generate",
        timeout: float = 120.0,
    ) -> None:
        self.model = model
        self.endpoint = endpoint
        self.timeout = timeout

    def generate(self, prompt: str) -> str:
        result = _post_json(
            self.endpoint,
            {"model": self.model, "prompt": prompt, "format": "json", "stream": False},
            {},
            self.timeout,
        )
        response = result.get("response")
        if not isinstance(response, str):
            raise TeacherError("Ollama response did not contain generated text")
        return response


class GeminiTeacher:
    """Call Gemini using an API key read from GEMINI_API_KEY."""

    name = "gemini"

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        api_key: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.timeout = timeout
        if not self.api_key:
            raise TeacherError("set GEMINI_API_KEY to use the Gemini teacher")

    def generate(self, prompt: str) -> str:
        endpoint = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent"
        )
        result = _post_json(
            endpoint,
            {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "responseMimeType": "application/json",
                    "temperature": 0.2,
                },
            },
            {"x-goog-api-key": self.api_key},
            self.timeout,
        )
        try:
            parts = result["candidates"][0]["content"]["parts"]
            response = "".join(part["text"] for part in parts if isinstance(part.get("text"), str))
        except (KeyError, IndexError, TypeError) as error:
            raise TeacherError("Gemini response did not contain generated text") from error
        if not response:
            raise TeacherError("Gemini returned empty generated text")
        return response