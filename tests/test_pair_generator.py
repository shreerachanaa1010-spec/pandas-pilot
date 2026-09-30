import json
from collections import Counter

from nlpandas.data.pair_generator import generate_pairs
from nlpandas.data.schema_generator import generate_dataset
from nlpandas.sandbox.runner import validate_code


def test_generate_pairs_verifies_templates_and_is_reproducible(tmp_path) -> None:
    input_dir = tmp_path / "raw"
    output_path = tmp_path / "processed" / "pairs.jsonl"
    generate_dataset(input_dir, schema_count=10, rows_per_schema=40, seed=19)

    report = generate_pairs(input_dir, output_path)
    first_output = output_path.read_bytes()
    second_report = generate_pairs(input_dir, output_path)

    examples = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    task_counts = Counter(example["task"] for example in examples)
    assert first_output == output_path.read_bytes()
    assert report == second_report
    assert report["candidate_count"] == report["accepted_count"] + report["rejected_count"]
    assert report["accepted_count"] == len(examples) > 0
    assert {
        "category_filter",
        "date_year_filter",
        "groupby_mean",
        "null_count",
        "string_search",
        "top_k",
    } <= set(task_counts)
    assert all(example["expected_output"] for example in examples)
    assert all(example["split"] in {"train", "validation", "test"} for example in examples)
    for example in examples:
        validate_code(example["code"])


def test_report_records_empty_or_invalid_candidate_rejections(tmp_path) -> None:
    input_dir = tmp_path / "raw"
    output_path = tmp_path / "processed" / "pairs.jsonl"
    generate_dataset(input_dir, schema_count=10, rows_per_schema=40, seed=4)

    report = generate_pairs(input_dir, output_path)
    report_file = output_path.with_name("pair_generation_report.json")

    assert json.loads(report_file.read_text(encoding="utf-8")) == report
    assert set(report["rejections"]) <= {"empty", "error", "non_deterministic"}