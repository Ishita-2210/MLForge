from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

from mlforge.config import Settings
from mlforge.errors import ForgeError
from mlforge.schemas import ExperimentConfig
from mlforge.service import Forge


def smoke(home: Path, task: str, algorithms: list[str] | None = None) -> dict:
    forge = Forge(Settings(home.resolve()))
    dataset = forge.sample(task)
    features = [s["name"] for s in dataset["schema"] if s["name"] not in ("target", "record_id")]
    config = ExperimentConfig(
        name=f"Synthetic {task} smoke",
        dataset_id=dataset["id"],
        target="target",
        task=task,
        features=features,
        models=algorithms or ["baseline", "linear", "tree", "forest"],
        primary_metric="f1_macro" if task == "classification" else "rmse",
    )
    t0 = time.perf_counter()
    exp = forge.start(config, background=False)
    if exp["status"] != "succeeded":
        raise ForgeError(exp.get("error", "Smoke training failed"))
    model = forge.register(exp["id"], f"synthetic-{task}")
    # A new service instance proves persistence/reload, rather than using the fitted object.
    reloaded = Forge(Settings(home.resolve()))
    frame = reloaded.data(dataset["id"]).frame
    row = (
        frame[features]
        .iloc[:2]
        .astype(object)
        .where(frame[features].iloc[:2].notna(), None)
        .to_dict("records")
    )
    response = reloaded.predict(model["id"], row)
    if len(response.predictions) != 2 or response.model_version != model["version"]:
        raise AssertionError("Smoke inference contract failed")
    drift = reloaded.drift_demo(model["id"])
    result = exp["result"]
    report = {
        "source": "deterministic synthetic data, seed 42",
        "task": task,
        "dataset_fingerprint": dataset["fingerprint"],
        "rows": dataset["rows"],
        "selected_algorithm": result["selected_algorithm"],
        "selection_source": result["selection_source"],
        "validation": result["candidates"][result["selected_index"]]["validation"],
        "final_holdout": result["final_test"],
        "final_test_evaluations": result["final_test_evaluations"],
        "experiment_id": exp["id"],
        "model_id": model["id"],
        "model_version": model["version"],
        "training_duration_s": result["duration_s"],
        "total_smoke_duration_s": time.perf_counter() - t0,
        "prediction_latency_ms": response.duration_ms,
        "artifact_bytes": model["bytes"],
        "memory": result["memory"],
        "candidates": [
            {
                k: c.get(k)
                for k in ("algorithm", "parameters", "status", "duration_s", "fit_s", "error")
            }
            for c in result["candidates"]
        ],
        "drift_demo_flagged_features": drift["drift"]["flagged_features"],
        "environment": result["environment"],
        "cpu": platform.processor(),
    }
    return report


def main() -> None:
    p = argparse.ArgumentParser(description="MLForge local dataset and reproducible smoke tools")
    p.add_argument("--home", type=Path, help="Local runtime-data directory")
    sub = p.add_subparsers(dest="command", required=True)
    smoke_parser = sub.add_parser("smoke")
    smoke_parser.add_argument(
        "--task", choices=["classification", "regression", "both"], default="both"
    )
    smoke_parser.add_argument("--output", type=Path)
    ingest_parser = sub.add_parser("ingest")
    ingest_parser.add_argument("csv", type=Path)
    sample_parser = sub.add_parser("sample")
    sample_parser.add_argument("--task", choices=["classification", "regression"], required=True)
    args = p.parse_args()
    home = args.home or Settings.default().home
    try:
        if args.command == "smoke":
            tasks = ["classification", "regression"] if args.task == "both" else [args.task]
            report = {task: smoke(home, task) for task in tasks}
            content = json.dumps(report, indent=2, allow_nan=False)
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(content + "\n", encoding="utf-8")
            print(content)
        elif args.command == "sample":
            print(json.dumps(Forge(Settings(home)).sample(args.task), indent=2))
        else:
            # Bound disk input before reading; service repeats validation for other callers.
            settings = Settings(home)
            if args.csv.stat().st_size > settings.max_bytes:
                raise ForgeError("CSV exceeds the configured byte limit")
            print(
                json.dumps(Forge(settings).ingest(args.csv.read_bytes(), args.csv.name), indent=2)
            )
    except (ForgeError, OSError) as exc:
        p.exit(2, f"MLForge error: {exc}\n")


if __name__ == "__main__":
    main()
