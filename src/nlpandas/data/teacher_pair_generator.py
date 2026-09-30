"""Generate teacher-written pairs and verify them through the sandbox."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from nlpandas.sandbox.runner import SandboxResult, run_code
from nlpandas.teachers.providers import GeminiTeacher, OllamaTeacher, Teacher, TeacherError

TASKS = (
    "category_filter",
    "groupby_mean",
    "top_k",
    "date_year_filter",
    "null_count",
    "string_search",
)
DIFFICULTIES = {"easy", "medium", "hard"}
MAX_QUESTION_LENGTH = 500
MAX_CODE_LENGTH = 16_000


@dataclass(frozen=True)
class TeacherExample:
    question: str
    code: str
    task: str
    difficulty: str


def parse_teacher_response(response: str, requested_task: str) -> TeacherExample:
    """Parse a bounded JSON response without executing any model-produced text."""
    text = response.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        if text.endswith("```"):
            text = text[:-3].strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError("teacher response is not valid JSON") from error
    if not isinstance(payload, dict):
        raise ValueError("teacher response must be a JSON object")

    question = payload.get("question")
    code = payload.get("code")
    task = payload.get("task")
    difficulty = payload.get("difficulty")
    if not isinstance(question, str) or not question.strip() or len(question) > MAX_QUESTION_LENGTH:
        raise ValueError("question must be a non-empty string of at most 500 characters")
    if not isinstance(code, str) or not code.strip() or len(code.encode("utf-8")) > MAX_CODE_LENGTH:
        raise ValueError("code must be non-empty and fit within the sandbox code limit")
    if task != requested_task or task not in TASKS:
        raise ValueError("teacher returned a task different from the requested task")
    if difficulty not in DIFFICULTIES:
        raise ValueError("difficulty must be easy, medium, or hard")
    return TeacherExample(question.strip(), code.strip(), task, difficulty)


def _json_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    clean_frame = frame.astype("object").where(pd.notna(frame), None)
    records = clean_frame.to_dict(orient="records")
    for record in records:
        for column, value in record.items():
            if hasattr(value, "item"):
                value = value.item()
            if hasattr(value, "isoformat"):
                value = value.isoformat()
            record[column] = value
    return records


def _build_prompt(schema: dict[str, Any], frame: pd.DataFrame, task: str) -> str:
    sample = _json_records(frame.head(8))
    return "\n".join(
        [
            "Write one natural-language question and its correct pandas solution.",
            "The DataFrame is named df and pandas is imported as pd.",
            "Assign the final answer to result. Do not import modules or access files, processes, "
            "or networks.",
            "Return only a JSON object with question, code, task, and difficulty fields.",
            "difficulty must be easy, medium, or hard; task must exactly match the requested task.",
            f"Requested task: {task}",
            f"Domain: {schema['domain']}",
            f"Columns: {json.dumps(schema['columns'])}",
            f"Sample rows: {json.dumps(sample, ensure_ascii=True, default=str)}",
        ]
    )


def generate_teacher_pairs(
    input_dir: Path,
    output_path: Path,
    teacher: Teacher,
    examples_per_schema: int = 10,
    schema_limit: int | None = None,
    sandbox: Callable[..., SandboxResult] = run_code,
) -> dict[str, Any]:
    """Generate train-split pairs only; sandbox each candidate twice before keeping it."""
    if examples_per_schema < 1:
        raise ValueError("examples_per_schema must be positive")
    if schema_limit is not None and schema_limit < 1:
        raise ValueError("schema_limit must be positive")

    manifest = json.loads((input_dir / "manifest.json").read_text(encoding="utf-8"))
    train_schemas = [schema for schema in manifest["schemas"] if schema["split"] == "train"]
    if schema_limit is not None:
        train_schemas = train_schemas[:schema_limit]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    accepted: list[dict[str, Any]] = []
    rejected: Counter[str] = Counter()
    accepted_by_task: Counter[str] = Counter()
    candidate_count = 0

    for schema in train_schemas:
        frame = pd.read_csv(input_dir / f"{schema['schema_id']}.csv")
        records = _json_records(frame)
        columns = list(frame.columns)
        for index in range(examples_per_schema):
            candidate_count += 1
            task = TASKS[index % len(TASKS)]
            try:
                raw_response = teacher.generate(_build_prompt(schema, frame, task))
                example = parse_teacher_response(raw_response, task)
            except (TeacherError, ValueError, TypeError):
                rejected["invalid_or_unavailable_teacher"] += 1
                continue

            first = sandbox(example.code, records, columns)
            if first.status != "ok" or first.output is None:
                rejected[first.status] += 1
                continue
            second = sandbox(example.code, records, columns)
            if second.status != "ok" or second.output is None:
                rejected[second.status] += 1
                continue
            if first.output != second.output:
                rejected["non_deterministic"] += 1
                continue

            if _empty_output(first.output):
                rejected["empty"] += 1
                continue
            accepted.append(
                {
                    "example_id": f"{schema['schema_id']}_teacher_{index:04d}",
                    "schema_id": schema["schema_id"],
                    "split": schema["split"],
                    "domain": schema["domain"],
                    **asdict(example),
                    "expected_output": first.output,
                    "teacher": teacher.name,
                }
            )
            accepted_by_task[task] += 1

    with output_path.open("w", encoding="utf-8", newline="\n") as output_file:
        for example in accepted:
            output_file.write(json.dumps(example, ensure_ascii=True, allow_nan=False) + "\n")

    report = {
        "teacher": teacher.name,
        "train_schemas_considered": len(train_schemas),
        "candidate_count": candidate_count,
        "accepted_count": len(accepted),
        "rejected_count": sum(rejected.values()),
        "rejections": dict(sorted(rejected.items())),
        "accepted_by_task": dict(sorted(accepted_by_task.items())),
    }
    output_path.with_name("teacher_pair_generation_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def _empty_output(output: dict[str, Any]) -> bool:
    kind = output.get("kind")
    if kind == "dataframe":
        return not output.get("rows")
    if kind == "series":
        values = output.get("values", [])
        return not values or all(value is None for value in values)
    return output.get("value") is None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=("ollama", "gemini"), required=True)
    parser.add_argument("--model")
    parser.add_argument("--input", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/teacher_pairs.jsonl"))
    parser.add_argument("--examples-per-schema", type=int, default=10)
    parser.add_argument("--schema-limit", type=int)
    args = parser.parse_args()

    if args.provider == "ollama":
        teacher = OllamaTeacher(model=args.model or "qwen2.5-coder:7b")
    else:
        teacher = GeminiTeacher(model=args.model or "gemini-2.5-flash")
    report = generate_teacher_pairs(
        args.input,
        args.output,
        teacher,
        examples_per_schema=args.examples_per_schema,
        schema_limit=args.schema_limit,
    )
    print(
        f"Accepted {report['accepted_count']}/{report['candidate_count']} teacher pairs; "
        f"rejected {report['rejected_count']}"
    )


if __name__ == "__main__":
    main()