from __future__ import annotations

import argparse
import json
import logging
import os
import threading
from pathlib import Path

import psutil
from filelock import FileLock, Timeout

from mlforge.artifacts import artifact_path, write_pipeline
from mlforge.config import Settings
from mlforge.errors import ForgeError
from mlforge.schemas import ExperimentConfig
from mlforge.service import Forge
from mlforge.training import environment, train

logger = logging.getLogger("mlforge.worker")


def execute(settings: Settings, key: str) -> None:
    forge = Forge(settings)
    store = forge.store
    try:
        with FileLock(str(settings.home / "training.lock"), timeout=0):
            exp = store.get("experiment", key)
            if exp["status"] == "cancelled" or exp.get("cancel_requested"):
                return
            store.patch(
                "experiment",
                key,
                status="running",
                pid=os.getpid(),
                process_created=psutil.Process().create_time(),
                progress="Training on CPU",
            )
            config = ExperimentConfig.model_validate(exp["config"])
            logger.info(
                json.dumps(
                    {
                        "event": "training_started",
                        "experiment_id": key,
                        "task": config.task,
                        "pid": os.getpid(),
                    }
                )
            )
            data = forge.data(config.dataset_id)
            pipeline, result = train(
                data,
                config,
                progress=lambda changes: store.patch("experiment", key, **changes),
                cancelled=lambda: store.get("experiment", key).get("cancel_requested", False),
            )
            artifact = write_pipeline(pipeline, artifact_path(settings.home, "experiments", key))
            artifact["environment"] = environment()
            artifact["artifact_path"] = f"experiments/{key}/pipeline.joblib"
            store.patch(
                "experiment",
                key,
                status="succeeded",
                result=result,
                artifact=artifact,
                candidates=result["candidates"],
                progress="Completed; final holdout evaluated once",
            )
            logger.info(
                json.dumps(
                    {
                        "event": "training_finished",
                        "experiment_id": key,
                        "status": "succeeded",
                        "duration_s": result["duration_s"],
                    }
                )
            )
    except Timeout:
        store.patch(
            "experiment", key, status="failed", error="Another training process holds the CPU lock"
        )
    except ForgeError as exc:
        store.patch(
            "experiment",
            key,
            status="cancelled" if exc.code == "cancelled" else "failed",
            error=str(exc),
            progress="Stopped",
        )
    except Exception as exc:
        store.patch(
            "experiment",
            key,
            status="failed",
            error=f"{type(exc).__name__}: training could not finish; inspect configuration and resource limits",
            progress="Failed",
        )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--home", type=Path, required=True)
    parser.add_argument("--id", required=True)
    args = parser.parse_args()
    settings = Settings(args.home.resolve())
    store = Forge(settings).store
    budget = store.get("experiment", args.id)["config"]["time_budget_s"]

    def expire():
        store.patch(
            "experiment",
            args.id,
            status="failed",
            error="Training exceeded the hard worker deadline",
            progress="Time budget exceeded",
        )
        os._exit(124)

    timer = threading.Timer(budget + 5, expire)
    timer.daemon = True
    timer.start()
    try:
        execute(settings, args.id)
    finally:
        timer.cancel()


if __name__ == "__main__":
    main()
