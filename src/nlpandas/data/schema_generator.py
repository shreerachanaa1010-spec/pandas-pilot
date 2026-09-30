"""Generate deterministic, lightly messy tabular schemas for evaluation."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from faker import Faker

DOMAIN_COLUMNS: dict[str, tuple[str, ...]] = {
    "sales": ("customer", "product", "region", "quantity", "amount", "event_date", "status"),
    "hr": ("employee", "department", "role", "salary", "region", "event_date", "status"),
    "logs": ("service", "level", "message", "response_ms", "region", "event_date", "status"),
    "finance": ("account", "transaction", "amount", "currency", "region", "event_date", "status"),
    "education": ("student", "course", "score", "instructor", "region", "event_date", "status"),
}
DOMAINS = tuple(DOMAIN_COLUMNS)
REGIONS = ("north", "south", "east", "west", "central")
STATUSES = ("new", "active", "pending", "closed", "cancelled")


@dataclass(frozen=True)
class DatasetSchema:
    """Metadata identifying one independently split synthetic table."""

    schema_id: str
    domain: str
    columns: tuple[str, ...]
    split: str


def assign_splits(schema_count: int, seed: int = 42) -> list[str]:
    """Assign schemas to 70/10/20 partitions without splitting rows by chance."""
    if schema_count < 10:
        raise ValueError("schema_count must be at least 10 for a meaningful 70/10/20 split")

    schema_indices = list(range(schema_count))
    random.Random(seed).shuffle(schema_indices)
    train_end = int(schema_count * 0.7)
    validation_end = int(schema_count * 0.8)
    splits = [""] * schema_count

    for index in schema_indices[:train_end]:
        splits[index] = "train"
    for index in schema_indices[train_end:validation_end]:
        splits[index] = "validation"
    for index in schema_indices[validation_end:]:
        splits[index] = "test"
    return splits


def _value_for(column: str, fake: Faker, rng: random.Random) -> Any:
    if column in {"customer", "employee", "instructor", "student"}:
        return fake.name()
    if column in {"product", "course", "service", "account"}:
        return fake.word().title()
    if column in {"department", "role", "transaction", "level", "currency"}:
        return fake.word().upper()
    if column in {"message"}:
        return fake.sentence(nb_words=5)
    if column == "region":
        return rng.choice(REGIONS)
    if column == "status":
        return rng.choice(STATUSES)
    if column in {"quantity", "response_ms"}:
        return rng.randint(1, 1200)
    if column == "score":
        return round(rng.uniform(35, 100), 1)
    if column in {"amount", "salary"}:
        return round(rng.uniform(20, 12000), 2)
    if column == "event_date":
        return fake.date_between(start_date="-5y", end_date="today").isoformat()
    return fake.word()


def generate_table(schema: DatasetSchema, rows: int, seed: int = 42) -> pd.DataFrame:
    """Create one repeatable table with nulls and occasional dtype noise."""
    if rows < 1:
        raise ValueError("rows must be positive")

    rng = random.Random(seed)
    fake = Faker()
    fake.seed_instance(seed)
    records = [
        {column: _value_for(column, fake, rng) for column in schema.columns}
        for _ in range(rows)
    ]
    frame = pd.DataFrame.from_records(records)
    if "record_id" in frame.columns:
        frame["record_id"] = range(rows)

    for column in schema.columns:
        if column in {"message", "record_id"}:
            continue
        if rng.random() < 0.75:
            null_count = max(1, rows // 50)
            for row_index in rng.sample(range(rows), min(null_count, rows)):
                frame.loc[row_index, column] = None

    numeric_column = next(
        (column for column in ("amount", "salary", "score", "quantity", "response_ms")
         if column in frame.columns),
        None,
    )
    if numeric_column is not None and rows >= 10:
        frame[numeric_column] = frame[numeric_column].astype("object")
        row_index = rng.randrange(rows)
        frame.loc[row_index, numeric_column] = "unknown"

    date_column = "event_date"
    if date_column in frame.columns and rows >= 10:
        frame.loc[rng.randrange(rows), date_column] = "not-a-date"

    return frame


def generate_dataset(
    output_dir: Path,
    schema_count: int = 50,
    rows_per_schema: int = 100,
    seed: int = 42,
) -> list[DatasetSchema]:
    """Write CSV tables and a JSON manifest, splitting at the schema level."""
    if rows_per_schema < 1:
        raise ValueError("rows_per_schema must be positive")

    output_dir.mkdir(parents=True, exist_ok=True)
    splits = assign_splits(schema_count, seed)
    schemas: list[DatasetSchema] = []

    for index, split in enumerate(splits):
        domain = DOMAINS[index % len(DOMAINS)]
        schema = DatasetSchema(
            schema_id=f"schema_{index:03d}",
            domain=domain,
            columns=("record_id", *DOMAIN_COLUMNS[domain]),
            split=split,
        )
        table = generate_table(schema, rows_per_schema, seed + index)
        table.to_csv(output_dir / f"{schema.schema_id}.csv", index=False)
        schemas.append(schema)

    manifest = {
        "seed": seed,
        "schema_count": schema_count,
        "rows_per_schema": rows_per_schema,
        "schemas": [asdict(schema) for schema in schemas],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return schemas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--schemas", type=int, default=50)
    parser.add_argument("--rows", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    schemas = generate_dataset(args.output, args.schemas, args.rows, args.seed)
    print(f"Wrote {len(schemas)} schemas to {args.output}")


if __name__ == "__main__":
    main()