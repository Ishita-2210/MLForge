"""Container entrypoint; local development continues to use scripts/dev.py."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    port = int(os.environ.get("PORT", "80"))
    if not 1 <= port <= 65535:
        raise SystemExit("PORT must be between 1 and 65535")
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(root / "streamlit_app.py"),
        "--server.address",
        "0.0.0.0",
        "--server.port",
        str(port),
        "--server.headless",
        "true",
        "--server.maxUploadSize",
        "2",
    ]
    hostname = os.environ.get("VERCEL_PROJECT_PRODUCTION_URL") or os.environ.get("VERCEL_URL")
    if hostname:
        command += ["--browser.serverAddress", hostname]
    os.chdir(root)
    os.execv(sys.executable, command)


if __name__ == "__main__":
    main()
