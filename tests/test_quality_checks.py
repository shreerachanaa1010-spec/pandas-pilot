import json

from nlpandas.data.quality_checks import audit_and_deduplicate


def _write_jsonl(path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )


def test_exact_pair_leakage_removes_training_rows_but_keeps_heldout(tmp_path) -> None:
    input_path = tmp_path / "pairs.jsonl"
    output_path = tmp_path / "curated.jsonl"
    rows = [
        {
            "example_id": "train-leak",
            "split": "train",
            "question": "Show one row",
            "code": "result=df.head(1)",
        },
        {
            "example_id": "train-duplicate",
            "split": "train",
            "question": "Show two rows",
            "code": "result = df.head(1)",
        },
        {
            "example_id": "train-clean",
            "split": "train",
            "question": "Show two rows",
            "code": "result = df.head(2)",
        },
        {
            "example_id": "train-clean-duplicate",
            "split": "train",
            "question": "Show two rows",
            "code": "result=df.head(2)",
        },
        {
            "example_id": "validation-reference",
            "split": "validation",
            "question": "Show one row",
            "code": "result = df.head(1)",
        },
        {
            "example_id": "test-only",
            "split": "test",
            "question": "Show three rows",
            "code": "result = df.head(3)",
        },
    ]
    _write_jsonl(input_path, rows)

    report = audit_and_deduplicate(input_path, output_path)
    curated = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]

    assert {row["example_id"] for row in curated} == {
        "train-clean",
        "train-duplicate",
        "validation-reference",
        "test-only",
    }
    assert report["exact_train_heldout_pair_hash_collisions"] == 1
    assert report["exact_train_duplicates"] == 1
    assert report["split_counts_after_curation"] == {"test": 1, "train": 2, "validation": 1}


def test_benchmark_exact_match_removes_only_training_copy(tmp_path) -> None:
    input_path = tmp_path / "pairs.jsonl"
    benchmark_path = tmp_path / "benchmark.jsonl"
    output_path = tmp_path / "curated.jsonl"
    _write_jsonl(
        input_path,
        [
            {
                "example_id": "train-benchmark-leak",
                "split": "train",
                "question": "Find maximum value",
                "code": "result = df['value'].max()",
            },
            {
                "example_id": "test-retained",
                "split": "test",
                "question": "A held-out question",
                "code": "result = df.head(1)",
            },
        ],
    )
    _write_jsonl(
        benchmark_path,
        [{"question": "Find maximum value", "code": "result=df['value'].max()"}],
    )

    report = audit_and_deduplicate(input_path, output_path, benchmark_path)
    curated = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]

    assert [row["example_id"] for row in curated] == ["test-retained"]
    assert report["benchmark_reference_count"] == 1
    assert report["exact_train_heldout_pair_hash_collisions"] == 1


def test_near_duplicate_is_flagged_and_not_automatically_removed(tmp_path) -> None:
    input_path = tmp_path / "pairs.jsonl"
    output_path = tmp_path / "curated.jsonl"
    _write_jsonl(
        input_path,
        [
            {
                "example_id": "train-near",
                "split": "train",
                "question": "Show maximum amount for each region",
                "code": "result = df['amount'].max()",
            },
            {
                "example_id": "test-near",
                "split": "test",
                "question": "Show minimum amount for each region",
                "code": "result = df['amount'].min()",
            },
        ],
    )

    report = audit_and_deduplicate(
        input_path, output_path, similarity_threshold=0.75
    )
    curated = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]

    assert len(curated) == 2
    assert report["near_duplicate_count"] == 1
    assert report["near_duplicates"][0]["training_id"] == "train-near"
    assert report["near_duplicates"][0]["reference_id"] == "test-near"