#!/bin/sh
set -eu

python3 - <<'PY'
import os
from pathlib import Path

answer = Path(os.environ.get("TASK_ROOT", "/app")) / "answer.txt"
logs = Path(os.environ.get("VERIFIER_LOG_DIR", "/logs/verifier"))
logs.mkdir(parents=True, exist_ok=True)
passed = answer.is_file() and answer.read_bytes() == b"Hello, Harbor!\n"
(logs / "reward.txt").write_text("1\n" if passed else "0\n", encoding="utf-8")
raise SystemExit(0 if passed else 1)
PY
