import json
from pathlib import Path

import pytest

from nlpandas.data.schema_generator import generate_dataset
from nlpandas.data.teacher_pair_generator import (
    TASKS,
    TeacherExample,
    generate_teacher_pairs,
    parse_teacher_response,
)
from nlpandas.sandbox.runner import SandboxResult
from nlpandas.teachers import providers
from nlpandas.teachers.providers import GeminiTeacher, OllamaTeacher


class FakeTeacher:
    name = "fake"

    def __init__(self, response_factory=None) -> None:
        self.prompts: list[str] = []
        self.response_factory = response_factory

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        task = next(line.removeprefix("Requested task: ") for line in prompt.splitlines()
                    if line.startswith("Requested task: "))
        if self.response_factory is not None:
            return self.response_factory(task, len(self.prompts))
        return json.dumps(
            {
                "question": f"Return one row for the {task} task.",
                "code": "result = df.head(1)",
                "task": task,
                "difficulty": "easy",
            }
        )


class FakeSandbox:
    def __init__(self, outputs=None, status: str = "ok") -> None:
        self.calls: list[tuple[str, list[dict], list[str]]] = []
        self.outputs = outputs or [{"kind": "dataframe", "columns": ["record_id"], "rows": [[0]]}]
        self.status = status

    def __call__(self, code: str, records: list[dict], columns: list[str]) -> SandboxResult:
        self.calls.append((code, records, columns))
        output = self.outputs[min(len(self.calls) - 1, len(self.outputs) - 1)]
        return SandboxResult(self.status, output=output)


def test_parse_teacher_json_and_markdown_fence() -> None:
    response = """```json
{"question":"Count rows","code":"result = df.head(1)","task":"top_k","difficulty":"easy"}
```"""

    result = parse_teacher_response(response, "top_k")

    assert result == TeacherExample("Count rows", "result = df.head(1)", "top_k", "easy")


def test_parse_teacher_response_rejects_wrong_task_and_malformed_json() -> None:
    with pytest.raises(ValueError, match="different from the requested task"):
        parse_teacher_response(
            json.dumps(
                {
                    "question": "q",
                    "code": "result = df.head(1)",
                    "task": "top_k",
                    "difficulty": "easy",
                }
            ),
            "category_filter",
        )
    with pytest.raises(ValueError, match="valid JSON"):
        parse_teacher_response("not json", "top_k")


def test_ollama_adapter_uses_local_json_endpoint(monkeypatch) -> None:
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self) -> bytes:
            return b'{"response":"generated text"}'

    def fake_urlopen(request, timeout):
        assert request.full_url == "http://localhost:11434/api/generate"
        assert timeout == 2
        assert json.loads(request.data)["model"] == "qwen-test"
        return FakeResponse()

    monkeypatch.setattr(providers, "urlopen", fake_urlopen)
    teacher = OllamaTeacher(model="qwen-test", timeout=2)

    assert teacher.generate("prompt") == "generated text"


def test_gemini_adapter_keeps_api_key_in_header(monkeypatch) -> None:
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self) -> bytes:
            return b'{"candidates":[{"content":{"parts":[{"text":"generated text"}]}}]}'

    def fake_urlopen(request, timeout):
        assert "secret-test-key" not in request.full_url
        assert request.get_header("X-goog-api-key") == "secret-test-key"
        assert (
            json.loads(request.data)["generationConfig"]["responseMimeType"]
            == "application/json"
        )
        return FakeResponse()

    monkeypatch.setattr(providers, "urlopen", fake_urlopen)
    teacher = GeminiTeacher(model="gemini-test", api_key="secret-test-key")

    assert teacher.generate("prompt") == "generated text"


def test_pipeline_only_prompts_train_schemas_and_sandboxes_twice(tmp_path: Path) -> None:
    input_dir = tmp_path / "raw"
    output_path = tmp_path / "processed" / "teacher_pairs.jsonl"
    schemas = generate_dataset(input_dir, schema_count=10, rows_per_schema=30, seed=11)
    teacher = FakeTeacher()
    sandbox = FakeSandbox()

    report = generate_teacher_pairs(
        input_dir,
        output_path,
        teacher,
        examples_per_schema=2,
        sandbox=sandbox,
    )

    examples = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    train_schema_ids = {schema.schema_id for schema in schemas if schema.split == "train"}
    assert report["train_schemas_considered"] == len(train_schema_ids) == 7
    assert report["candidate_count"] == report["accepted_count"] == 14
    assert len(teacher.prompts) == 14
    assert len(sandbox.calls) == 28
    assert {example["split"] for example in examples} == {"train"}
    assert {example["schema_id"] for example in examples} <= train_schema_ids
    assert [example["task"] for example in examples[: len(TASKS)]] == list(TASKS[:2]) * 3


def test_pipeline_rejects_unsafe_or_non_deterministic_results(tmp_path: Path) -> None:
    input_dir = tmp_path / "raw"
    generate_dataset(input_dir, schema_count=10, rows_per_schema=30, seed=13)
    teacher = FakeTeacher()
    sandbox = FakeSandbox(status="rejected")

    report = generate_teacher_pairs(
        input_dir,
        tmp_path / "unsafe.jsonl",
        teacher,
        examples_per_schema=1,
        schema_limit=1,
        sandbox=sandbox,
    )

    assert report["candidate_count"] == 1
    assert report["accepted_count"] == 0
    assert report["rejections"] == {"rejected": 1}
    assert len(sandbox.calls) == 1

    changing_sandbox = FakeSandbox(
        outputs=[
            {"kind": "scalar", "value": 1},
            {"kind": "scalar", "value": 2},
        ]
    )
    unstable_report = generate_teacher_pairs(
        input_dir,
        tmp_path / "unstable.jsonl",
        teacher,
        examples_per_schema=1,
        schema_limit=1,
        sandbox=changing_sandbox,
    )
    assert unstable_report["rejections"] == {"non_deterministic": 1}