"""Launch the local Streamlit UI and FastAPI, then stop owned children on Ctrl+C."""

from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]


def free_port(port: int) -> None:
    with socket.socket() as sock:
        try:
            sock.bind(("127.0.0.1", port))
        except OSError as exc:
            raise SystemExit(
                f"Port {port} is already occupied. Stop that process or choose another port."
            ) from exc


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ui-port", type=int, default=8501)
    p.add_argument("--api-port", type=int, default=8000)
    a = p.parse_args()
    if a.ui_port == a.api_port or not all(1024 <= n <= 65535 for n in (a.ui_port, a.api_port)):
        p.error("Choose distinct ports between 1024 and 65535")
    free_port(a.ui_port)
    free_port(a.api_port)
    env = dict(
        os.environ,
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",
        MLFORGE_API_URL=f"http://127.0.0.1:{a.api_port}",
    )
    commands = [
        [
            sys.executable,
            "-m",
            "uvicorn",
            "app.api:create_app",
            "--factory",
            "--host",
            "127.0.0.1",
            "--port",
            str(a.api_port),
        ],
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "streamlit_app.py",
            "--server.address",
            "127.0.0.1",
            "--server.port",
            str(a.ui_port),
            "--server.headless",
            "true",
        ],
    ]
    processes = []
    try:
        for command in commands:
            processes.append(subprocess.Popen(command, cwd=ROOT, env=env))
        print(
            f"MLForge UI: http://127.0.0.1:{a.ui_port}\nPrediction API: http://127.0.0.1:{a.api_port}/docs\nCtrl+C stops both services.",
            flush=True,
        )
        while all(process.poll() is None for process in processes):
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        owned = []
        for process in processes:
            try:
                parent = psutil.Process(process.pid)
                owned.extend(parent.children(recursive=True))
                owned.append(parent)
            except psutil.Error:
                pass
        for process in reversed(owned):
            try:
                process.terminate()
            except psutil.Error:
                pass
        _, live = psutil.wait_procs(owned, timeout=5)
        for process in live:
            try:
                process.kill()
            except psutil.Error:
                pass


if __name__ == "__main__":
    main()
