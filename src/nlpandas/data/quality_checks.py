"""Deduplicate exact training leakage and report near-duplicate data pairs."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON on line {line_number} of {path}") from error
        if not isinstance(record, dict):
            raise ValueError(f"line {line_number} of {path} must be a JSON object")
        if not isinstance(record.get("question"), str) or not isinstance(record.get("code"), str):
            raise ValueError(f"line {line_number} of {path} needs string question and code fields")
        records.append(record)
    return records


def _normalize_question(question: str) -> str:
    return " ".join(re.findall(r"\w+", question.casefold()))


def _canonical_code(code: str) -> str:
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError:
        return " ".join(code.split())
    return ast.dump(tree, annotate_fields=True, include_attributes=False)


def _fingerprint(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _pair_fingerprint(record: dict[str, Any]) -> str:
    normalized_pair = (
        f"{_normalize_question(record['question'])}\0{_canonical_code(record['code'])}"
    )
    return _fingerprint(normalized_pair)


def _record_id(record: dict[str, Any], index: int) -> str:
    value = record.get("example_id")
    return value if isinstance(value, str) else f"row-{index}"


def _near_duplicate_report(
    training: list[tuple[int, dict[str, Any]]],
    references: list[tuple[str, dict[str, Any]]],
    threshold: float,
) -> list[dict[str, Any]]:
    if not training or not references:
        return []
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.neighbors import NearestNeighbors
    except ImportError as error:
        raise RuntimeError(
            "install the optional quality dependencies with `python -m pip install -e .[quality]`"
        ) from error

    def text(record: dict[str, Any]) -> str:
        return f"{_normalize_question(record['question'])}\n{_canonical_code(record['code'])}"

    reference_texts = [text(record) for _, record in references]
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
    reference_vectors = vectorizer.fit_transform(reference_texts)
    training_vectors = vectorizer.transform([text(record) for _, record in training])
    neighbors = NearestNeighbors(n_neighbors=1, metric="cosine", algorithm="brute")
    neighbors.fit(reference_vectors)
    distances, indices = neighbors.kneighbors(training_vectors)

    near_duplicates: list[dict[str, Any]] = []
    for training_position, (distance, reference_position) in enumerate(
        zip(distances[:, 0], indices[:, 0], strict=True)
    ):
        similarity = 1.0 - float(distance)
        if similarity >= threshold:
            training_index, training_record = training[training_position]
            reference_id, reference_record = references[int(reference_position)]
            near_duplicates.append(
                {
                    "training_id": _record_id(training_record, training_index),
                    "reference_id": reference_id,
                    "reference_split": reference_record.get("split", "benchmark"),
                    "similarity": round(similarity, 6),
                }
            )
    return near_duplicates


def audit_and_deduplicate(
    input_path: Path,
    output_path: Path,
    benchmark_path: Path | None = None,
    similarity_threshold: float = 0.92,
) -> dict[str, Any]:
    """Remove exact train leakage and write an auditable curation report.

    Validation/test rows are never removed. TF-IDF near matches are reported for
    review but retained; only exact normalized code-hash collisions are pruned
    from training automatically.
    """
    if not 0 < similarity_threshold <= 1:
        raise ValueError("similarity_threshold must be between 0 and 1")

    records = _read_jsonl(input_path)
    benchmark_records = _read_jsonl(benchmark_path) if benchmark_path else []
    indexed_records = list(enumerate(records))
    heldout = [
        (index, record)
        for index, record in indexed_records
        if record.get("split") in {"validation", "test"}
    ]
    external = [
        (f"benchmark-{index}", record) for index, record in enumerate(benchmark_records)
    ]
    references = [
        (f"{_record_id(record, index)}", record) for index, record in heldout
    ] + external
    reference_pair_hashes = {_pair_fingerprint(record) for _, record in references}

    training = [
        (index, record)
        for index, record in indexed_records
        if record.get("split") == "train"
    ]
    removed_indices: set[int] = set()
    exact_code_leaks: list[str] = []
    exact_pair_duplicates: list[str] = []
    train_pair_seen: set[tuple[str, str]] = set()

    for index, record in training:
        question = _normalize_question(record["question"])
        canonical_code = _canonical_code(record["code"])
        pair_key = (question, canonical_code)
        pair_hash = _pair_fingerprint(record)
        record_id = _record_id(record, index)
        if pair_hash in reference_pair_hashes:
            removed_indices.add(index)
            exact_code_leaks.append(record_id)
        elif pair_key in train_pair_seen:
            removed_indices.add(index)
            exact_pair_duplicates.append(record_id)
        train_pair_seen.add(pair_key)

    filtered_training = [item for item in training if item[0] not in removed_indices]
    near_duplicates = _near_duplicate_report(filtered_training, references, similarity_threshold)

    curated = [record for index, record in indexed_records if index not in removed_indices]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as output_file:
        for record in curated:
            output_file.write(json.dumps(record, ensure_ascii=True, allow_nan=False) + "\n")

    split_counts = Counter(str(record.get("split", "unspecified")) for record in curated)
    report: dict[str, Any] = {
        "input_count": len(records),
        "output_count": len(curated),
        "removed_training_count": len(removed_indices),
        "exact_train_heldout_pair_hash_collisions": len(exact_code_leaks),
        "exact_train_duplicates": len(exact_pair_duplicates),
        "removed_training_ids": sorted(exact_code_leaks + exact_pair_duplicates),
        "near_duplicate_threshold": similarity_threshold,
        "near_duplicate_count": len(near_duplicates),
        "near_duplicates": near_duplicates,
        "split_counts_after_curation": dict(sorted(split_counts.items())),
        "benchmark_reference_count": len(benchmark_records),
    }
    report_path = output_path.with_name("quality_report.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/processed/pairs.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/curated_pairs.jsonl"))
    parser.add_argument("--benchmark", type=Path)
    parser.add_argument("--similarity-threshold", type=float, default=0.92)
    args = parser.parse_args()
    report = audit_and_deduplicate(
        args.input, args.output, args.benchmark, args.similarity_threshold
    )
    print(
        f"Curated {report['output_count']}/{report['input_count']} pairs; "
        f"removed {report['removed_training_count']} training duplicates; "
        f"flagged {report['near_duplicate_count']} near-duplicates"
    )


if __name__ == "__main__":
    main()