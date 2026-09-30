"""Build and verify starter NL-to-pandas pairs from generated tables.

Only fixed code templates authored in this module are executed here. Do not use
this module to execute teacher output or other untrusted code; that belongs in
the isolated sandbox stage.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

NUMERIC_COLUMNS = ("amount", "salary", "score", "quantity", "response_ms")
TEXT_COLUMNS: dict[str, tuple[str, ...]] = {
    "sales": ("customer", "product"),
    "hr": ("employee", "department", "role"),
    "logs": ("service", "message", "level"),
    "finance": ("account", "transaction", "currency"),
    "education": ("student", "course", "instructor"),
}


@dataclass(frozen=True)
class Candidate:
    """One internally generated question and its pandas solution template."""

    schema_id: str
    split: str
    domain: str
    task: str
    difficulty: str
    question: str
    code: str


def _literal(value: Any) -> str:
    """Return a Python literal for a value read from our generated CSV."""
    if isinstance(value, str):
        return repr(value)
    return repr(value.item() if hasattr(value, "item") else value)


def _build_candidates(schema: dict[str, Any], frame: pd.DataFrame) -> list[Candidate]:
    domain = schema["domain"]
    schema_id = schema["schema_id"]
    split = schema["split"]
    candidates: list[Candidate] = []

    status_value = frame["status"].dropna().iloc[0]
    candidates.append(
        Candidate(
            schema_id,
            split,
            domain,
            "category_filter",
            "easy",
            f"Show all rows where status is {_literal(status_value)}.",
            f"result = df.loc[df['status'].eq({_literal(status_value)})].copy()",
        )
    )

    numeric_column = next(column for column in NUMERIC_COLUMNS if column in frame.columns)
    candidates.extend(
        [
            Candidate(
                schema_id,
                split,
                domain,
                "groupby_mean",
                "medium",
                f"For each region, calculate the mean {numeric_column}, ignoring invalid values.",
                "\n".join(
                    [
                        f"numeric = pd.to_numeric(df[{numeric_column!r}], errors='coerce')",
                        "result = numeric.groupby(df['region'], dropna=False).mean()"
                        ".dropna().sort_values(ascending=False).head(5)",
                    ]
                ),
            ),
            Candidate(
                schema_id,
                split,
                domain,
                "top_k",
                "medium",
                f"Show the five rows with the highest valid {numeric_column} values.",
                "\n".join(
                    [
                        f"numeric = pd.to_numeric(df[{numeric_column!r}], errors='coerce')",
                        "result = df.loc[numeric.nlargest(5).index].copy()",
                    ]
                ),
            ),
        ]
    )

    dates = pd.to_datetime(frame["event_date"], errors="coerce")
    valid_years = dates.dropna().dt.year
    if not valid_years.empty:
        year = int(valid_years.iloc[0])
        candidates.append(
            Candidate(
                schema_id,
                split,
                domain,
                "date_year_filter",
                "medium",
                f"Show rows from {year}, treating invalid dates as missing.",
                "\n".join(
                    [
                        "dates = pd.to_datetime(df['event_date'], errors='coerce')",
                        f"result = df.loc[dates.dt.year.eq({year})].copy()",
                    ]
                ),
            )
        )

    null_columns = [
        column
        for column in frame.columns
        if column != "record_id" and frame[column].isna().any()
    ]
    if null_columns:
        column = null_columns[0]
        candidates.append(
            Candidate(
                schema_id,
                split,
                domain,
                "null_count",
                "easy",
                f"How many values are missing from {column}?",
                f"result = int(df[{column!r}].isna().sum())",
            )
        )

    text_column = next(
        (column for column in TEXT_COLUMNS[domain] if column in frame.columns), None
    )
    if text_column is not None:
        text_values = frame[text_column].dropna()
        if not text_values.empty:
            search_value = str(text_values.iloc[0])
            candidates.append(
                Candidate(
                    schema_id,
                    split,
                    domain,
                    "string_search",
                    "easy",
                    f"Find rows where {text_column} contains {_literal(search_value)}.",
                    "result = df.loc["
                    f"df[{text_column!r}].astype('string').str.contains("
                    f"{_literal(search_value)}, case=False, regex=False, na=False)"
                    "].copy()",
                )
            )

    return candidates


def _json_value(value: Any) -> Any:
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _normalize_result(result: Any) -> dict[str, Any]:
    if isinstance(result, pd.DataFrame):
        return {
            "kind": "dataframe",
            "columns": [str(column) for column in result.columns],
            "rows": [
                [_json_value(value) for value in row]
                for row in result.itertuples(index=False, name=None)
            ],
        }
    if isinstance(result, pd.Series):
        return {
            "kind": "series",
            "name": None if result.name is None else str(result.name),
            "index": [_json_value(value) for value in result.index.tolist()],
            "values": [_json_value(value) for value in result.tolist()],
        }
    return {"kind": "scalar", "value": _json_value(result)}


def _has_result(result: dict[str, Any]) -> bool:
    if result["kind"] == "dataframe":
        return bool(result["rows"])
    if result["kind"] == "series":
        return bool(result["values"]) and any(value is not None for value in result["values"])
    return result["value"] is not None


def _execute_template(candidate: Candidate, frame: pd.DataFrame) -> dict[str, Any]:
    """Execute only code produced by _build_candidates in a fresh namespace."""
    namespace: dict[str, Any] = {"df": frame.copy(), "pd": pd}
    safe_globals = {"__builtins__": {"int": int}}
    exec(compile(candidate.code, "<trusted-pandas-template>", "exec"), safe_globals, namespace)
    return _normalize_result(namespace["result"])


def _verify_candidate(
    candidate: Candidate, frame: pd.DataFrame
) -> tuple[dict[str, Any] | None, str | None]:
    try:
        first = _execute_template(candidate, frame)
        if not _has_result(first):
            return None, "empty"
        second = _execute_template(candidate, frame)
        if first != second:
            return None, "non_deterministic"
        json.dumps(first, allow_nan=False)
        return first, None
    except Exception:
        return None, "error"


def generate_pairs(input_dir: Path, output_path: Path) -> dict[str, int]:
    """Generate JSONL pairs and a report of accepted and rejected templates."""
    manifest_path = input_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    accepted: list[dict[str, Any]] = []
    rejected: Counter[str] = Counter()
    task_counts: Counter[str] = Counter()

    for schema in manifest["schemas"]:
        frame = pd.read_csv(input_dir / f"{schema['schema_id']}.csv")
        for candidate in _build_candidates(schema, frame):
            expected_output, rejection = _verify_candidate(candidate, frame)
            if rejection is not None:
                rejected[rejection] += 1
                continue
            accepted.append(
                {
                    "example_id": f"{candidate.schema_id}_{candidate.task}",
                    "schema_id": candidate.schema_id,
                    "split": candidate.split,
                    "domain": candidate.domain,
                    "task": candidate.task,
                    "difficulty": candidate.difficulty,
                    "question": candidate.question,
                    "code": candidate.code,
                    "expected_output": expected_output,
                }
            )
            task_counts[candidate.task] += 1

    with output_path.open("w", encoding="utf-8", newline="\n") as output_file:
        for example in accepted:
            output_file.write(json.dumps(example, ensure_ascii=True, allow_nan=False) + "\n")

    report = {
        "candidate_count": len(accepted) + sum(rejected.values()),
        "accepted_count": len(accepted),
        "rejected_count": sum(rejected.values()),
        "rejections": dict(sorted(rejected.items())),
        "accepted_by_task": dict(sorted(task_counts.items())),
    }
    report_path = output_path.with_name("pair_generation_report.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/pairs.jsonl"))
    args = parser.parse_args()
    report = generate_pairs(args.input, args.output)
    print(
        f"Accepted {report['accepted_count']}/{report['candidate_count']} pairs; "
        f"rejected {report['rejected_count']}"
    )


if __name__ == "__main__":
    main()