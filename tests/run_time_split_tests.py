"""
run_time_split_tests.py — Convenience wrapper that runs the predictive-validation smoke tests.

Usage:  python tests/run_time_split_tests.py
"""
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

rc = subprocess.run(
    [sys.executable, str(PROJECT_ROOT / "tests" / "test_time_splits.py")],
    cwd=str(PROJECT_ROOT),
).returncode

sys.exit(rc)
