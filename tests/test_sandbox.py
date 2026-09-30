import shutil

import pytest

from nlpandas.sandbox.runner import SandboxConfig, _docker_command, run_code, validate_code

ATTACKS = [
    "while True: pass",
    "result = [0] * (10 ** 9)",
    "result = open('/etc/passwd').read()",
    "import urllib.request",
    "import os\nos.fork()",
]


@pytest.mark.parametrize("code", ATTACKS)
def test_adversarial_code_is_rejected_before_container_launch(monkeypatch, code) -> None:
    def fail_if_launched(*args, **kwargs):
        pytest.fail("rejected code must never reach Docker")

    monkeypatch.setattr("nlpandas.sandbox.runner.shutil.which", lambda _: "docker")
    monkeypatch.setattr("nlpandas.sandbox.runner.subprocess.run", fail_if_launched)

    result = run_code(code, records=[{"value": 1}], columns=["value"])

    assert result.status == "rejected"


def test_supported_pandas_subset_passes_static_validation() -> None:
    validate_code('result = df.loc[df["status"].eq("active")].copy()')


def test_docker_command_enforces_isolation_and_resource_limits() -> None:
    command = _docker_command(SandboxConfig(), "test-sandbox")

    assert "--network=none" in command
    assert "--pull=never" in command
    assert "--read-only" in command
    assert "--cap-drop=ALL" in command
    assert "--memory=256m" in command
    assert "--cpus=0.5" in command
    assert "--pids-limit=32" in command
    assert "--user=10001:10001" in command


def test_missing_docker_is_reported_without_running_code(monkeypatch) -> None:
    monkeypatch.setattr("nlpandas.sandbox.runner.shutil.which", lambda _: None)

    result = run_code("result = df.head(1)", records=[{"value": 1}], columns=["value"])

    assert result.status == "unavailable"
    assert "Docker CLI" in result.message


@pytest.mark.skipif(
    shutil.which("docker") is None, reason="Docker is required for integration test"
)
def test_valid_pandas_code_runs_in_isolated_container() -> None:
    result = run_code(
        'result = df.loc[df["value"].eq(2)].copy()',
        records=[{"value": 1}, {"value": 2}],
        columns=["value"],
    )

    assert result.status == "ok"
    assert result.output == {"kind": "dataframe", "columns": ["value"], "rows": [[2]]}