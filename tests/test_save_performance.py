"""Exercise the actual save script with a fake git, including failed pushes."""
import os
from pathlib import Path
import subprocess

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "save-performance.sh"


def execute(tmp_path, failures=0, message="remote: Internal Server Error", unchanged=False):
    fake = tmp_path / "git"
    fake.write_text('''#!/usr/bin/env python3
import os, pathlib, sys
root = pathlib.Path(os.environ["FAKE_GIT_ROOT"])
with (root / "calls").open("a") as f:
    f.write(" ".join(sys.argv[1:]) + "\\n")
if sys.argv[1] == "diff":
    sys.exit(0 if os.environ["UNCHANGED"] == "1" else 1)
if sys.argv[1] == "push":
    p = root / "attempts"
    attempt = int(p.read_text()) + 1 if p.exists() else 1
    p.write_text(str(attempt))
    if attempt <= int(os.environ["FAILURES"]):
        print(os.environ["FAILURE_MESSAGE"], file=sys.stderr)
        sys.exit(1)
sys.exit(0)
''', encoding="utf-8")
    fake.chmod(0o755)
    env = dict(os.environ, PATH=f"{tmp_path}:{os.environ['PATH']}",
               FAKE_GIT_ROOT=str(tmp_path), FAILURES=str(failures), FAILURE_MESSAGE=message,
               UNCHANGED="1" if unchanged else "0", SAVE_RETRY_DELAY_SECONDS="0")
    result = subprocess.run(["bash", str(SCRIPT)], cwd=tmp_path, env=env,
                            text=True, capture_output=True, timeout=10)
    return result, (tmp_path / "calls").read_text().splitlines()


def test_unchanged_records_do_not_commit_or_push(tmp_path):
    result, calls = execute(tmp_path, unchanged=True)
    assert result.returncode == 0
    assert not any(c.startswith(("commit", "push")) for c in calls)


def test_internal_server_error_retries_the_same_commit(tmp_path):
    result, calls = execute(tmp_path, failures=2)
    assert result.returncode == 0
    assert calls.count("push origin HEAD:main") == 3
    assert len([c for c in calls if c.startswith("commit")]) == 1
    assert not any("--force" in c for c in calls)


def test_exhaustion_is_a_visible_failure_not_silent_success(tmp_path):
    result, calls = execute(tmp_path, failures=10)
    assert result.returncode == 1
    assert calls.count("push origin HEAD:main") == 4
    assert "backed up as an artifact" in result.stdout


@pytest.mark.parametrize("message", ["Permission denied", "403 Forbidden", "[rejected] non-fast-forward"])
def test_permissions_and_conflicts_are_not_retried(tmp_path, message):
    result, calls = execute(tmp_path, failures=10, message=message)
    assert result.returncode == 1
    assert calls.count("push origin HEAD:main") == 1
