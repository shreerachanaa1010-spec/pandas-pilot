import json
from collections import Counter

import pandas as pd
import pytest

from nlpandas.data.schema_generator import assign_splits, generate_dataset


def test_assign_splits_is_deterministic_and_schema_level() -> None:
    first = assign_splits(50, seed=17)

    assert first == assign_splits(50, seed=17)
    assert Counter(first) == {"train": 35, "validation": 5, "test": 10}


def test_generate_dataset_writes_reproducible_tables_and_manifest(tmp_path) -> None:
    output_dir = tmp_path / "generated"
    schemas = generate_dataset(output_dir, schema_count=10, rows_per_schema=20, seed=7)
    first_csv = (output_dir / "schema_000.csv").read_bytes()

    generate_dataset(output_dir, schema_count=10, rows_per_schema=20, seed=7)

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    table = pd.read_csv(output_dir / "schema_000.csv")
    assert (output_dir / "schema_000.csv").read_bytes() == first_csv
    assert len(schemas) == len(manifest["schemas"]) == 10
    assert len(table) == 20
    assert table["record_id"].tolist() == list(range(20))
    assert Counter(schema.split for schema in schemas) == {
        "train": 7,
        "validation": 1,
        "test": 2,
    }


def test_assign_splits_rejects_too_few_schemas() -> None:
    with pytest.raises(ValueError, match="at least 10"):
        assign_splits(9)