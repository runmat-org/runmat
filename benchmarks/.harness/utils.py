import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

MSE_REGEX = re.compile(r"RESULT_ok\s+MSE\s*=\s*([0-9eE+\-.]+)")

def which(cmd: str) -> Optional[str]:
    return shutil.which(cmd)

def run_cmd(cmd: List[str], cwd: Optional[Path] = None, env: Optional[Dict[str, str]] = None, timeout: int = 0) -> Tuple[int, float, str, str]:
    start = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            text=True,
            timeout=timeout if timeout and timeout > 0 else None,
        )
        rc = proc.returncode
        stdout = proc.stdout
        stderr = proc.stderr
    except subprocess.TimeoutExpired as e:
        rc = 124
        stdout = e.stdout or ""
        stderr = (e.stderr or "") + "\nTIMEOUT"
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return rc, elapsed_ms, stdout, stderr


def parse_mse(text: str) -> Optional[float]:
    m = MSE_REGEX.search(text)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def median(values: List[float]) -> float:
    s = sorted(values)
    n = len(s)
    if n == 0:
        return float("nan")
    mid = n // 2
    if n % 2 == 1:
        return s[mid]
    return 0.5 * (s[mid - 1] + s[mid])


def ensure_output_path(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def default_env() -> Dict[str, str]:
    env = os.environ.copy()
    # Favor headless/quiet modes where relevant
    env.setdefault("NO_GUI", "1")
    env.setdefault("RUNMAT_PLOT_MODE", "headless")
    return env