"""Run a narrow subset of pandas code inside a resource-limited Docker container."""

from __future__ import annotations

import ast
import json
import math
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SandboxConfig:
    image: str = "nl-pandas-sandbox:latest"
    timeout_seconds: float = 5.0
    max_code_bytes: int = 16_000
    max_payload_bytes: int = 4_000_000
    max_output_bytes: int = 1_000_000
    max_rows: int = 20_000
    max_columns: int = 200
    memory_limit: str = "256m"
    cpu_limit: str = "0.5"
    pids_limit: int = 32


@dataclass(frozen=True)
class SandboxResult:
    status: str
    output: dict[str, Any] | None = None
    message: str | None = None
    elapsed_ms: float = 0.0


ALLOWED_NAMES = {"df", "pd", "result", "numeric", "dates", "int"}
ALLOWED_ATTRIBUTES = {
    "astype",
    "contains",
    "copy",
    "dropna",
    "dt",
    "eq",
    "groupby",
    "head",
    "index",
    "isna",
    "loc",
    "mean",
    "month",
    "nlargest",
    "sort_values",
    "str",
    "sum",
    "to_datetime",
    "to_numeric",
    "year",
}
ALLOWED_NODES = {
    ast.Module,
    ast.Assign,
    ast.Name,
    ast.Load,
    ast.Store,
    ast.Constant,
    ast.Subscript,
    ast.Attribute,
    ast.Call,
    ast.keyword,
    ast.Compare,
    ast.BinOp,
    ast.BitAnd,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.In,
    ast.NotIn,
    ast.Is,
    ast.IsNot,
    ast.BoolOp,
    ast.And,
    ast.Or,
    ast.UnaryOp,
    ast.Not,
    ast.USub,
    ast.UAdd,
    ast.List,
    ast.Tuple,
    ast.Dict,
    ast.Slice,
}
ASSIGNABLE_NAMES = {"result", "numeric", "dates"}

WORKER_SCRIPT = r'''
import json
import math
import sys
import pandas as pd

def normalize(value):
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)

def normalize_result(value):
    if isinstance(value, pd.DataFrame):
        return {
            "kind": "dataframe",
            "columns": [str(column) for column in value.columns],
            "rows": [[normalize(item) for item in row]
                     for row in value.itertuples(index=False, name=None)],
        }
    if isinstance(value, pd.Series):
        return {
            "kind": "series",
            "name": None if value.name is None else str(value.name),
            "index": [normalize(item) for item in value.index.tolist()],
            "values": [normalize(item) for item in value.tolist()],
        }
    return {"kind": "scalar", "value": normalize(value)}

try:
    payload = json.loads(sys.stdin.read())
    frame = pd.DataFrame.from_records(payload["records"], columns=payload["columns"])
    namespace = {"df": frame, "pd": pd}
    safe_globals = {"__builtins__": {"int": int}}
    exec(compile(payload["code"], "<generated-pandas>", "exec"), safe_globals, namespace)
    sys.stdout.write(json.dumps(normalize_result(namespace["result"]), allow_nan=False))
except Exception as error:
    sys.stderr.write(type(error).__name__ + ": " + str(error)[:500])
    sys.exit(2)
'''


class CodeRejected(ValueError):
    """Raised when code is outside the supported pandas subset."""


def validate_code(code: str, config: SandboxConfig | None = None) -> None:
    """Reject unsupported syntax, names, attributes, and oversized literals."""
    settings = config or SandboxConfig()
    if not code.strip():
        raise CodeRejected("code is empty")
    if len(code.encode("utf-8")) > settings.max_code_bytes:
        raise CodeRejected("code exceeds the configured size limit")

    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as error:
        raise CodeRejected("code is not valid Python syntax") from error

    for node in ast.walk(tree):
        if type(node) not in ALLOWED_NODES:
            raise CodeRejected(f"unsupported syntax: {type(node).__name__}")
        if isinstance(node, ast.Name) and node.id not in ALLOWED_NAMES:
            raise CodeRejected(f"unsupported name: {node.id}")
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("_") or node.attr not in ALLOWED_ATTRIBUTES:
                raise CodeRejected(f"unsupported attribute: {node.attr}")
        if isinstance(node, ast.Call):
            function = node.func
            if isinstance(function, ast.Name):
                allowed = function.id == "int"
            else:
                allowed = isinstance(function, ast.Attribute) and (
                    function.attr in ALLOWED_ATTRIBUTES
                )
            if not allowed:
                raise CodeRejected("unsupported function call")
        if isinstance(node, ast.Assign):
            if any(
                not isinstance(target, ast.Name) or target.id not in ASSIGNABLE_NAMES
                for target in node.targets
            ):
                raise CodeRejected("assignments must target result or an approved temporary")
        if isinstance(node, ast.Constant):
            value = node.value
            if isinstance(value, str) and len(value) > 256:
                raise CodeRejected("string literal exceeds the configured size limit")
            if isinstance(value, int) and not isinstance(value, bool) and abs(value) > 1_000_000:
                raise CodeRejected("integer literal exceeds the configured limit")
            if isinstance(value, float) and not math.isfinite(value):
                raise CodeRejected("non-finite numeric literal is not allowed")


def _docker_command(config: SandboxConfig, container_name: str) -> list[str]:
    return [
        "docker",
        "run",
        "--rm",
        "--interactive",
        "--pull=never",
        f"--name={container_name}",
        "--network=none",
        f"--memory={config.memory_limit}",
        f"--memory-swap={config.memory_limit}",
        f"--cpus={config.cpu_limit}",
        f"--pids-limit={config.pids_limit}",
        "--read-only",
        "--tmpfs=/tmp:rw,noexec,nosuid,size=16m",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--user=10001:10001",
        "--ipc=none",
        "--ulimit=cpu=3:3",
        "--ulimit=fsize=1048576:1048576",
        "--ulimit=nofile=64:64",
        config.image,
        "python",
        "-I",
        "-c",
        WORKER_SCRIPT,
    ]


def _empty_result(output: dict[str, Any]) -> bool:
    kind = output.get("kind")
    if kind == "dataframe":
        return not output.get("rows")
    if kind == "series":
        values = output.get("values", [])
        return not values or all(value is None for value in values)
    return output.get("value") is None


def run_code(
    code: str,
    records: list[dict[str, Any]],
    columns: list[str],
    config: SandboxConfig | None = None,
) -> SandboxResult:
    """Execute generated pandas code in a Docker container with no network."""
    settings = config or SandboxConfig()
    started = time.perf_counter()
    try:
        validate_code(code, settings)
    except CodeRejected as error:
        return SandboxResult("rejected", message=str(error))

    if len(records) > settings.max_rows or len(columns) > settings.max_columns:
        return SandboxResult("rejected", message="table exceeds the configured size limit")
    if not columns or any(not isinstance(column, str) for column in columns):
        return SandboxResult("rejected", message="columns must be non-empty strings")
    if any(not isinstance(record, dict) or not set(record).issubset(columns) for record in records):
        return SandboxResult("rejected", message="records must be mappings of declared columns")

    payload = {"code": code, "records": records, "columns": columns}
    try:
        serialized_payload = json.dumps(payload, allow_nan=False, ensure_ascii=True)
    except (TypeError, ValueError):
        return SandboxResult("rejected", message="table contains non-JSON or non-finite values")
    if len(serialized_payload.encode("utf-8")) > settings.max_payload_bytes:
        return SandboxResult("rejected", message="input exceeds the configured size limit")

    docker = shutil.which("docker")
    if docker is None:
        return SandboxResult("unavailable", message="Docker CLI is not installed")

    container_name = f"nlpandas-sandbox-{uuid.uuid4().hex}"
    command = _docker_command(settings, container_name)
    command[0] = docker
    try:
        process = subprocess.run(
            command,
            input=serialized_payload,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=settings.timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired:
        try:
            subprocess.run(
                [docker, "rm", "--force", container_name],
                capture_output=True,
                timeout=3,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            pass
        elapsed = (time.perf_counter() - started) * 1000
        return SandboxResult(
            "timeout", message="execution exceeded its time limit", elapsed_ms=elapsed
        )
    except OSError as error:
        return SandboxResult("unavailable", message=str(error))

    elapsed = (time.perf_counter() - started) * 1000
    if len(process.stdout.encode("utf-8")) > settings.max_output_bytes:
        return SandboxResult(
            "rejected", message="output exceeds the configured size limit", elapsed_ms=elapsed
        )
    if process.returncode != 0:
        message = process.stderr[:1000].strip() or "container execution failed"
        return SandboxResult("execution_error", message=message, elapsed_ms=elapsed)

    try:
        output = json.loads(process.stdout)
    except json.JSONDecodeError:
        return SandboxResult(
            "execution_error", message="container returned invalid JSON", elapsed_ms=elapsed
        )
    if _empty_result(output):
        return SandboxResult("empty", output=output, elapsed_ms=elapsed)
    return SandboxResult("ok", output=output, elapsed_ms=elapsed)